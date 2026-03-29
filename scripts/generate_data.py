# -*- coding: utf-8 -*-
"""
生成简单的算术数据集，用于验证想法

数据格式和COCONUT一致:
{
    "question": "3 + 5 = ?",
    "answer": "8",
    "steps": ["3加5", "等于8"]
}
"""

import json
import random
import os

def generate_addition(a, b):
    """加法问题"""
    return {
        "question": f"{a} + {b} = ?",
        "answer": str(a + b),
        "steps": [f"{a}加{b}", f"等于{a+b}"]
    }

def generate_subtraction(a, b):
    """减法问题 (保证结果非负)"""
    if a < b:
        a, b = b, a
    return {
        "question": f"{a} - {b} = ?",
        "answer": str(a - b),
        "steps": [f"{a}减{b}", f"等于{a-b}"]
    }

def generate_multiplication(a, b):
    """乘法问题"""
    return {
        "question": f"{a} × {b} = ?",
        "answer": str(a * b),
        "steps": [f"{a}乘{b}", f"等于{a*b}"]
    }

def generate_two_step():
    """两步运算 (需要更多思考)"""
    a, b, c = random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)
    op1 = random.choice(['+', '-'])
    op2 = random.choice(['+', '-'])

    # 计算中间结果
    if op1 == '+':
        temp = a + b
    else:
        temp = abs(a - b)

    # 计算最终结果
    if op2 == '+':
        result = temp + c
    else:
        result = abs(temp - c)

    op1_sym = '+' if op1 == '+' else '-'
    op2_sym = '+' if op2 == '+' else '-'

    return {
        "question": f"{a} {op1_sym} {b} {op2_sym} {c} = ?",
        "answer": str(result),
        "steps": [
            f"先算{a}{op1_sym}{b}={temp}",
            f"再算{temp}{op2_sym}{c}={result}"
        ]
    }

def generate_three_step():
    """三步运算 (需要更多思考)"""
    nums = [random.randint(1, 5) for _ in range(4)]
    ops = random.choices(['+', '-'], k=3)

    # 从左到右计算
    result = nums[0]
    steps = []
    for i, op in enumerate(ops):
        prev = result
        if op == '+':
            result = result + nums[i+1]
        else:
            result = abs(result - nums[i+1])
        op_sym = '+' if op == '+' else '-'
        steps.append(f"第{i+1}步: {prev}{op_sym}{nums[i+1]}={result}")

    # 构建问题字符串
    op_syms = ['+' if o == '+' else '-' for o in ops]
    question = f"{nums[0]} {op_syms[0]} {nums[1]} {op_syms[1]} {nums[2]} {op_syms[2]} {nums[3]} = ?"

    return {
        "question": question,
        "answer": str(result),
        "steps": steps
    }

def generate_dataset(
    output_dir,
    train_size=500,
    val_size=100,
    seed=42
):
    """
    生成完整数据集

    难度分布:
    - 30% 一步加法 (简单)
    - 20% 一步减法 (简单)
    - 20% 一步乘法 (中等)
    - 20% 两步运算 (中等)
    - 10% 三步运算 (困难)
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    def generate_samples(n):
        samples = []
        for _ in range(n):
            r = random.random()
            if r < 0.3:
                # 简单加法
                a, b = random.randint(1, 20), random.randint(1, 20)
                samples.append(generate_addition(a, b))
            elif r < 0.5:
                # 简单减法
                a, b = random.randint(1, 20), random.randint(1, 20)
                samples.append(generate_subtraction(a, b))
            elif r < 0.7:
                # 乘法
                a, b = random.randint(1, 10), random.randint(1, 10)
                samples.append(generate_multiplication(a, b))
            elif r < 0.9:
                # 两步
                samples.append(generate_two_step())
            else:
                # 三步
                samples.append(generate_three_step())
        return samples

    train_data = generate_samples(train_size)
    val_data = generate_samples(val_size)

    # 保存
    train_path = os.path.join(output_dir, "train.json")
    val_path = os.path.join(output_dir, "val.json")

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"生成完成!")
    print(f"  训练集: {len(train_data)} 样本 -> {train_path}")
    print(f"  验证集: {len(val_data)} 样本 -> {val_path}")

    # 打印几个例子
    print("\n样本示例:")
    for i in range(3):
        print(f"\n--- 样本 {i+1} ---")
        sample = train_data[i]
        print(f"问题: {sample['question']}")
        print(f"步骤: {sample['steps']}")
        print(f"答案: {sample['answer']}")

    return train_data, val_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed
    )
