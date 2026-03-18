import gc
import re
import random
from typing import List, Dict, Optional, Tuple
import json, os
import argparse
from kubernetes import client, config
from langchain_community.llms import Ollama
import difflib
import torch
from prompt import namespace
from transformers import AutoTokenizer, AutoModelForCausalLM

from RL_config import hf_cache_path, bnb_config, get_response_from_llm, get_reward_from_llm_response, read_mop_file

###############################################
# Generate IO Pairs for DPO Training
# Input data for DPO should be in the form of
# {"prompt": ..., "chosen": ..., "rejected": ...}
# Not CoT! Orginal IP pairs code
###############################################
def generate_io_pairs(
    model_path_or_id: str,
    mop_data: List[Dict[str, str]],
    k8s_client,
    form : str = 'Python',
    max_new_tokens: int = 500,
    temperature: float = 0.3,
    top_p: float = 0.8,
    device: Optional[str] = None,
    steps: int = 5,
) -> List[Dict[str, str]]:
    """
    MOP data (list[dict[str, str]])를 받아 LLM 출력 생성 후,
    [{"input": ..., "chosen": ..., "rejected": ...}, ...] 형태로 반환.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    gc.collect()
    torch.cuda.empty_cache()
    v1, apps_v1 = k8s_client
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_id, cache_dir=hf_cache_path)
    # decoder-only 모델의 padding 문제 방지
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path_or_id, quantization_config=bnb_config,device_map="auto",cache_dir=hf_cache_path)
    model.eval()

    results: List[Dict[str, str]] = []
    vm_num = {}
    # 배치로 처리하고 싶으면 batch_size를 추가로 받아서 chunking하면 됨.
    model_name = model_path_or_id.split('/')[-1]
    for step_num in range(steps):
        for single_mop_data in mop_data:
            prompt = single_mop_data['mop']
            vnf = single_mop_data['vnf']
            if vnf not in vm_num:
                vm_num[vnf] = 1
            else:
                vm_num[vnf] += 1
            lang = single_mop_data['lang']
            vm_name = single_mop_data['vm_name']
            print ('\n'+'#'*50)
            print (f'## Generating IO pairs for VNF: {vnf}, Language: {lang}, VM: {vm_name}, MOP:')
            #print(prompt)
            #print('\n ## Now here is answer')
            answer = get_response_from_llm(model, tokenizer, max_new_tokens, temperature, top_p, prompt, do_sample=True)
            #print(answer)

            # Test in NDT
            success, _ = get_reward_from_llm_response(answer, form, vnf, model_name, vm_num, k8s_client, namespace)
            if success == 1.0:
                good_config_answer = answer
                print('## Good configuration example generated successfully.')
                bad_config_answer = None
                for try_num in range(5):  # 최대 5번까지 시도
                    answer = get_response_from_llm(model, tokenizer, max_new_tokens, min(0.9, temperature+0.1*(try_num+1)), top_p, prompt, do_sample=True)
                    success, _ = get_reward_from_llm_response(answer, form, vnf, model_name, vm_num, k8s_client, namespace)
                    if success != 1.0:
                        bad_config_answer = answer
                        break
                if bad_config_answer is None:
                    print('## Failed to generate bad configuration example after multiple attempts, skipping this pair.')
                else:
                    print('## Bad configuration example generated successfully, added to IO pairs.')
                    results.append({"input": prompt, "chosen": good_config_answer, "rejected": bad_config_answer})
    print (f'Total {len(results)} IO pairs generated for DPO training.')
    return results

def build_cot_prompt(base_prompt: str, style: str = "strong") -> str:
    """
    style:
        - strong: reasoning + final
        - verified: reasoning + verification + final
        - direct: final only
    """
    if style == "strong":
        instruction = """
You must solve the task with concise but clear reasoning.

Output format exactly:
<REASONING>
Step-by-step reasoning here.
</REASONING>
<FINAL>
Final answer only.
</FINAL>

Rules:
- Put all reasoning inside <REASONING> ... </REASONING>
- Put the deployable/configurable final answer only inside <FINAL> ... </FINAL>
- The <FINAL> block must be directly executable/usable as the answer
- Do not omit the <FINAL> block
"""
    elif style == "verified":
        instruction = """
You must solve the task with reasoning and a short verification step.

Output format exactly:
<REASONING>
Step-by-step reasoning here.
Verification: briefly check constraints / correctness.
</REASONING>
<FINAL>
Final answer only.
</FINAL>

Rules:
- Put all reasoning inside <REASONING> ... </REASONING>
- Include a short verification/check step in reasoning
- Put the deployable/configurable final answer only inside <FINAL> ... </FINAL>
- The <FINAL> block must be directly executable/usable as the answer
- Do not omit the <FINAL> block
"""
    elif style == "direct":
        instruction = """
Output format exactly:
<FINAL>
Final answer only.
</FINAL>

Rules:
- Do not include reasoning
- The <FINAL> block must be directly executable/usable as the answer
"""
    else:
        raise ValueError(f"Unknown style: {style}")

    return f"{instruction}\n\nTask:\n{base_prompt}"


# =========================
# Parsing helpers
# =========================
REASONING_PATTERN = re.compile(r"<REASONING>\s*(.*?)\s*</REASONING>", re.DOTALL | re.IGNORECASE)
FINAL_PATTERN = re.compile(r"<FINAL>\s*(.*?)\s*</FINAL>", re.DOTALL | re.IGNORECASE)


def parse_reasoning_and_final(text: str) -> Tuple[str, str]:
    reasoning_match = REASONING_PATTERN.search(text or "")
    final_match = FINAL_PATTERN.search(text or "")

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    final = final_match.group(1).strip() if final_match else ""

    # fallback:
    # FINAL 블록이 없으면 전체를 final로 간주하는 fallback을 둘 수도 있음
    if not final:
        final = text.strip()

    return reasoning, final


def format_cot_answer(reasoning: str, final: str) -> str:
    reasoning = reasoning.strip()
    final = final.strip()

    if reasoning:
        return f"<REASONING>\n{reasoning}\n</REASONING>\n<FINAL>\n{final}\n</FINAL>"
    return f"<FINAL>\n{final}\n</FINAL>"


def format_direct_answer(final: str) -> str:
    final = final.strip()
    return f"<FINAL>\n{final}\n</FINAL>"


# =========================
# Reasoning transformation helpers
# =========================
def split_reasoning_steps(reasoning: str) -> List[str]:
    """
    매우 단순한 step 분리.
    """
    reasoning = reasoning.strip()
    if not reasoning:
        return []

    # 줄 단위 우선
    lines = [line.strip() for line in reasoning.splitlines() if line.strip()]
    if len(lines) >= 2:
        return lines

    # 문장 단위 fallback
    sentences = re.split(r"(?<=[.!?])\s+", reasoning)
    return [s.strip() for s in sentences if s.strip()]


def contains_verification(reasoning: str) -> bool:
    keywords = [
        "verify", "verification", "check", "checked", "validate", "validated",
        "constraint", "constraints", "sanity", "test", "confirmed",
        "검증", "확인", "체크"
    ]
    lower_text = reasoning.lower()
    return any(k in lower_text for k in keywords)


def remove_verification_steps(reasoning: str) -> str:
    steps = split_reasoning_steps(reasoning)
    filtered = []

    verification_keywords = [
        "verify", "verification", "check", "checked", "validate", "validated",
        "constraint", "constraints", "sanity", "test", "confirmed",
        "검증", "확인", "체크"
    ]

    for step in steps:
        s = step.lower()
        if any(k in s for k in verification_keywords):
            continue
        filtered.append(step)

    if not filtered:
        # 최소한 reasoning이 완전히 비지 않게 원문 일부 유지
        filtered = steps[:max(1, len(steps) // 2)]

    return "\n".join(filtered).strip()


def weaken_reasoning(reasoning: str) -> str:
    """
    strong CoT -> weak CoT로 자동 열화
    전략:
      - step 일부 제거
      - verification 제거
      - 너무 길면 앞/뒤 핵심 일부만 남김
    """
    steps = split_reasoning_steps(reasoning)
    if not steps:
        return ""

    # verification 먼저 제거
    steps = [s for s in steps if s.strip()]
    if not steps:
        return reasoning.strip()

    weak_steps = []
    for step in steps:
        s = step.lower()
        if any(k in s for k in ["verify", "verification", "check", "validate", "검증", "확인", "체크"]):
            continue
        weak_steps.append(step)

    if not weak_steps:
        weak_steps = steps[:]

    # step을 줄여서 논리 점프 유도
    if len(weak_steps) >= 4:
        # 첫 단계 + 마지막 단계 정도만 유지
        weak_steps = [weak_steps[0], weak_steps[-1]]
    elif len(weak_steps) == 3:
        weak_steps = [weak_steps[0], weak_steps[-1]]
    elif len(weak_steps) == 2:
        weak_steps = [weak_steps[-1]]

    weakened = "\n".join(weak_steps).strip()

    # 너무 짧으면 원본 일부를 남김
    if not weakened:
        weakened = weak_steps[0].strip() if weak_steps else reasoning.strip()

    return weakened


# =========================
# Candidate generation / evaluation
# =========================
def generate_candidate(
    model,
    model_way,
    tokenizer,
    base_prompt: str,
    style: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, str]:
    full_prompt = build_cot_prompt(base_prompt, style=style)
    if model_way == 'Ollama':
        raw_output = model.invoke(full_prompt)
    else:
        raw_output = get_response_from_llm(
            model,
            tokenizer,
            max_new_tokens,
            temperature,
            top_p,
            full_prompt,
            do_sample=True
        )
    reasoning, final = parse_reasoning_and_final(raw_output)

    return {
        "style": style,
        "raw_output": raw_output,
        "reasoning": reasoning,
        "final": final,
        "formatted": format_cot_answer(reasoning, final) if reasoning else format_direct_answer(final),
    }


def evaluate_final_answer(
    final_answer: str,
    form: str,
    vnf: str,
    model_name: str,
    vm_num: Dict[str, int],
    k8s_client,
    namespace: str,
) -> Tuple[float, str]:
    """
    reward는 FINAL 블록만 평가
    """
    return get_reward_from_llm_response(final_answer, form, vnf, model_name, vm_num, k8s_client, namespace, use_whole_code=True)


def score_strong_cot_candidate(candidate: Dict[str, str]) -> int:
    """
    heuristic
    """
    reasoning = candidate.get("reasoning", "").strip()
    final = candidate.get("final", "").strip()

    if not final:
        return -999

    score = 0
    steps = split_reasoning_steps(reasoning)

    if reasoning:
        score += 2
    if len(steps) >= 2:
        score += 2
    if len(steps) >= 3:
        score += 1
    if contains_verification(reasoning):
        score += 2
    # 너무 과도하게 긴 reasoning은 약간 감점
    if len(reasoning) > 3000:
        score -= 1
    return score


def select_best_passing_candidate(candidates: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    if not candidates:
        return None
    return sorted(candidates, key=score_strong_cot_candidate, reverse=True)[0]


def select_best_failing_cot_candidate(candidates: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """
    wrong CoT용 negative:
    - reasoning이 있어야 하고
    - 너무 엉망인 direct fail보다 CoT fail 선호
    """
    cot_candidates = [c for c in candidates if c.get("reasoning", "").strip()]
    if not cot_candidates:
        return None
    return sorted(cot_candidates, key=score_strong_cot_candidate, reverse=True)[0]

def print_word_diff(text1, text2):
    # 단어 단위로 리스트 생성
    words1 = text1.split()
    words2 = text2.split()
    
    s = difflib.SequenceMatcher(None, words1, words2)
    
    # 완전히 일치하는지 먼저 확인
    if s.ratio() == 1.0:
        print("✅ 다른 부분이 없습니다.")
        return

    print("🔍 [변경 사항 요약]")
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            continue
        
        # 각 상태에 따라 한 줄로 출력
        a_part = " ".join(words1[i1:i2])
        b_part = " ".join(words2[j1:j2])
        
        if tag == 'replace':
            print(f"🔄 교체: '{a_part}' -> '{b_part}'")
        elif tag == 'delete':
            print(f"❌ 삭제: '{a_part}'")
        elif tag == 'insert':
            print(f"➕ 추가: '{b_part}'")


def save_cot_data(cot_data_list, filename):
    """
    Saves CoT data to a JSON file.
    If existing data is present, reads it and appends the new data.
    
    :param cot_data_list: List of CoT data to add (e.g., [{"prompt": "...", "chain_of_thought": "..."}, ...])
    :param filename: Path to the JSON file to save
    """
    # Read existing data (initialize as empty list if file does not exist)
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {filename} file is corrupted or invalid JSON. Creating a new one.")
            existing_data = []
    else:
        existing_data = []
    
    # Append new data
    existing_data.extend(cot_data_list)
    
    # Save to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)
    
    print(f"CoT data saved to {filename}. Total {len(existing_data)} items.")


# =========================
# Main function
# =========================
def generate_cot_dpo_pairs(
    model_path_or_id: str,
    model_way: str,
    mop_data: List[Dict[str, str]],
    k8s_client,  # Kubernetes client
    namespace: str,
    form: str = "Python",
    max_new_tokens: int = 800,
    temperature: float = 0.4,
    top_p: float = 0.9,
    device: Optional[str] = None,
    steps: int = 2,
    samples_per_prompt: int = 6,
    target_ratios: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """
    CoT-aware DPO dataset 생성

    출력 예:
    [
        {
            "input": ...,
            "chosen": ...,
            "rejected": ...,
            "category": "direct_vs_strong"
        },
        ...
    ]
    """

    if target_ratios is None:
        target_ratios = {
            "direct_vs_strong": 0.30,
            "weak_vs_strong": 0.30,
            "wrong_vs_correct": 0.20,
            "verified_vs_unverified": 0.20,
        }

    random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    v1, apps_v1 = k8s_client
    if model_way == 'Ollama':
        model = Ollama(model=model_path_or_id, num_ctx=131072)
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path_or_id,
            cache_dir=hf_cache_path
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_id,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=hf_cache_path
        )
        model.eval()
    if model_way== 'Ollama':
        model_name = model_path_or_id.split(":")[0]
    else:
        model_name = model_path_or_id.split("/")[-1]
    vm_num: Dict[str, int] = {}

    direct_vs_strong_pairs: List[Dict[str, str]] = []
    weak_vs_strong_pairs: List[Dict[str, str]] = []
    wrong_vs_correct_pairs: List[Dict[str, str]] = []
    verified_vs_unverified_pairs: List[Dict[str, str]] = []
    success_num=0
    for step_num in range(steps):
        print(f"\n{'=' * 80}")
        print(f"## Global step: {step_num + 1}/{steps}")

        for idx, single_mop_data in enumerate(mop_data):
            prompt = single_mop_data["mop"]
            vnf = single_mop_data["vnf"]
            lang = single_mop_data["lang"]
            vm_name = single_mop_data["vm_name"]

            if vnf not in vm_num:
                vm_num[vnf] = 1
            else:
                vm_num[vnf] += 1

            print("\n" + "#" * 60)
            print(f"## [{idx+1}/{len(mop_data)}] VNF={vnf}, LANG={lang}, VM={vm_name}")

            passing_candidates = []
            failing_candidates = []

            # 1) 후보 여러 개 생성
            generation_styles = ["strong", "verified", "strong", "verified", "strong", "direct"]
            if samples_per_prompt < len(generation_styles):
                generation_styles = generation_styles[:samples_per_prompt]
            elif samples_per_prompt > len(generation_styles):
                generation_styles += random.choices(["strong", "verified", "direct"], k=samples_per_prompt - len(generation_styles))
            for sample_idx, style in enumerate(generation_styles):
                cur_temp = min(0.95, temperature + 0.05 * sample_idx)

                candidate = generate_candidate(
                    model=model,
                    model_way=model_way,
                    tokenizer=tokenizer,
                    base_prompt=prompt,
                    style=style,
                    max_new_tokens=max_new_tokens,
                    temperature=cur_temp,
                    top_p=top_p,
                )
                #print_word_diff(before_answer, candidate["final"])
                #before_answer = candidate["final"]
                # reward는 FINAL만 평가
                success, reward_detail = evaluate_final_answer(
                    final_answer=candidate["final"],
                    form=form,
                    vnf=vnf,
                    model_name=model_name,
                    vm_num=vm_num,
                    k8s_client=k8s_client,
                    namespace=namespace,
                )
                #print (candidate["final"])
                print (f"Reward: {success} | Detail: {reward_detail}")
                candidate["success"] = success
                candidate["reward_detail"] = reward_detail

                if success == 1.0:
                    passing_candidates.append(candidate)
                    print(f"  - PASS | style={style} | score={score_strong_cot_candidate(candidate)}")
                else:
                    failing_candidates.append(candidate)
                    print(f"  - FAIL | style={style} | score={score_strong_cot_candidate(candidate)}")

            # 2) strong chosen 선정
            best_pass = select_best_passing_candidate(passing_candidates)
            if best_pass is None:
                print("## No passing candidate found. Skip this sample.")
                continue

            strong_reasoning = best_pass["reasoning"]
            strong_final = best_pass["final"]
            strong_formatted = format_cot_answer(strong_reasoning, strong_final)

            print("## Best passing candidate selected.")
            success_num+=1
            # =========================================================
            # A. direct answer vs strong CoT
            # =========================================================
            direct_answer = format_direct_answer(strong_final)
            if direct_answer.strip() != strong_formatted.strip():
                direct_vs_strong_pairs.append({
                    "input": prompt,
                    "chosen": strong_formatted,
                    "rejected": direct_answer,
                    "category": "direct_vs_strong",
                })

            # =========================================================
            # B. weak CoT vs strong CoT
            # =========================================================
            weak_reasoning = weaken_reasoning(strong_reasoning)
            weak_formatted = format_cot_answer(weak_reasoning, strong_final)

            if weak_reasoning.strip() and weak_formatted.strip() != strong_formatted.strip():
                weak_vs_strong_pairs.append({
                    "input": prompt,
                    "chosen": strong_formatted,
                    "rejected": weak_formatted,
                    "category": "weak_vs_strong",
                })

            # =========================================================
            # C. wrong CoT vs correct CoT
            # =========================================================
            best_fail = select_best_failing_cot_candidate(failing_candidates)
            if best_fail is not None:
                wrong_formatted = format_cot_answer(best_fail["reasoning"], best_fail["final"])
                if wrong_formatted.strip() != strong_formatted.strip():
                    wrong_vs_correct_pairs.append({
                        "input": prompt,
                        "chosen": strong_formatted,
                        "rejected": wrong_formatted,
                        "category": "wrong_vs_correct",
                    })

            # =========================================================
            # D. verified CoT vs unverified CoT
            # =========================================================
            if contains_verification(strong_reasoning):
                unverified_reasoning = remove_verification_steps(strong_reasoning)
                unverified_formatted = format_cot_answer(unverified_reasoning, strong_final)

                if unverified_reasoning.strip() and unverified_formatted.strip() != strong_formatted.strip():
                    verified_vs_unverified_pairs.append({
                        "input": prompt,
                        "chosen": strong_formatted,
                        "rejected": unverified_formatted,
                        "category": "verified_vs_unverified",
                    })
            else:
                # best_pass에 verification이 없으면 verified 스타일 pass 후보를 추가로 탐색
                verified_passes = [
                    c for c in passing_candidates
                    if contains_verification(c.get("reasoning", ""))
                ]
                if verified_passes:
                    best_verified = select_best_passing_candidate(verified_passes)
                    chosen_verified = format_cot_answer(best_verified["reasoning"], best_verified["final"])
                    unverified_reasoning = remove_verification_steps(best_verified["reasoning"])
                    rejected_unverified = format_cot_answer(unverified_reasoning, best_verified["final"])

                    if rejected_unverified.strip() != chosen_verified.strip():
                        verified_vs_unverified_pairs.append({
                            "input": prompt,
                            "chosen": chosen_verified,
                            "rejected": rejected_unverified,
                            "category": "verified_vs_unverified",
                        })
            if idx%20==19:
                print(f"\n{idx+1}th work end. total {success_num} camdidates created.")
                category_map = {
                    "direct_vs_strong": direct_vs_strong_pairs,
                    "weak_vs_strong": weak_vs_strong_pairs,
                    "wrong_vs_correct": wrong_vs_correct_pairs,
                    "verified_vs_unverified": verified_vs_unverified_pairs,
                }
                with open(f'tmp/cot_dpo_candidates_{system_name}_{form}.json', 'w') as f:
                    json.dump(category_map, f, indent=2)
    # =========================================================
    # Ratio balancing
    # =========================================================
    category_map = {
        "direct_vs_strong": direct_vs_strong_pairs,
        "weak_vs_strong": weak_vs_strong_pairs,
        "wrong_vs_correct": wrong_vs_correct_pairs,
        "verified_vs_unverified": verified_vs_unverified_pairs,
    }

    for cat, pairs in category_map.items():
        random.shuffle(pairs)
        print(f"{cat}: {len(pairs)} pairs")

    # 가능한 최대 공통 크기 기준으로 ratio 맞춤
    # target_count * ratio = 각 카테고리 샘플 수
    max_total_candidates = sum(len(v) for v in category_map.values())
    if max_total_candidates == 0:
        print("No CoT DPO pairs generated.")
        return []

    feasible_totals = []
    for cat, ratio in target_ratios.items():
        if ratio <= 0:
            continue
        feasible_totals.append(int(len(category_map[cat]) / ratio))

    target_total = min(feasible_totals) if feasible_totals else 0
    if target_total <= 0:
        # fallback: 그냥 전부 합침
        final_results = []
        for pairs in category_map.values():
            final_results.extend(pairs)
        random.shuffle(final_results)
        print(f"Total {len(final_results)} CoT DPO pairs generated (fallback merge).")
        return final_results

    final_results: List[Dict[str, str]] = []
    for cat, ratio in target_ratios.items():
        n = int(target_total * ratio)
        final_results.extend(category_map[cat][:n])

    random.shuffle(final_results)

    print(f"Total {len(final_results)} CoT DPO pairs generated.")
    print(f"  - direct_vs_strong: {sum(1 for x in final_results if x['category'] == 'direct_vs_strong')}")
    print(f"  - weak_vs_strong: {sum(1 for x in final_results if x['category'] == 'weak_vs_strong')}")
    print(f"  - wrong_vs_correct: {sum(1 for x in final_results if x['category'] == 'wrong_vs_correct')}")
    print(f"  - verified_vs_unverified: {sum(1 for x in final_results if x['category'] == 'verified_vs_unverified')}")

    return final_results

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--Ansible', action='store_true')
    argparser.add_argument('--Python', action='store_true')
    argparser.add_argument('--test', action='store_true', help='Test with small number of MOPs for quick run')
    argparser=argparser.parse_args()
    mop_file_path = '../data_generating/data_v3/'
    system_name='Kubernetes'
    config.load_kube_config()
    v1 = client.CoreV1Api()
    apps_v1 = client.AppsV1Api()
    form='Python' if argparser.Python else 'Ansible'
    mop_data = read_mop_file(mop_file_path,system_name, test=argparser.test)
    model_name = 'qwen3.5:35b'
    model_way = 'Ollama'
    k8s_client = (v1, apps_v1)
    generated_pairs = generate_cot_dpo_pairs(
        model_path_or_id=model_name,
        model_way=model_way,
        mop_data=mop_data,
        k8s_client=k8s_client,
        namespace=namespace,
        form=form,
        max_new_tokens=800)
    save_cot_data(generated_pairs, f'cot_dpo_pairs_{system_name}_{form}.json')