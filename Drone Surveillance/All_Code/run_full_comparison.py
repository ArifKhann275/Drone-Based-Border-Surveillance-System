# -*- coding: utf-8 -*-
"""
run_full_comparison.py
══════════════════════════════════════════════════════════════
C (RelayScore) vs D (true FIS) vs D2 (surrogate FIS) vs T (TOPSIS) vs
V (VIKOR) -- সবগুলো একসাথে, একই condition-এ (default terrain, static
wind, 8-drone fleet -- thesis-এর headline condition)। এটাই thesis-এ
add করার মতো মূল তুলনার টেবিল/ফলাফল তৈরি করে।

আগে থেকে যা লাগবে (একই ফোল্ডারে):
    a20.py
    a2e_relay_fixed_final_51.py   (এইমাত্র আপডেট করা ভার্সন)
    policy_d_true_fis.py
    surrogate_fis.py
    luts/                          (তোমার আগে বানানো LUT ফাইল সহ)
    run_full_comparison.py         (এই ফাইলটা)

চালানোর নিয়ম:
    python run_full_comparison.py

    কিছু না দিলে ডিফল্ট 50 seed, 500 steps চালাবে। দ্রুত টেস্ট করতে:
    python run_full_comparison.py 60 5   (max_steps=60, seeds=5)

ফলাফল কোথায় পাবে:
    - টার্মিনালে সরাসরি: প্রতিটা metric-এ mean, Wilcoxon p-value, effect
      size (rank-biserial r) -- প্রতিটা policy-জোড়ার জন্য
    - শেষে "Headline: selector time & speedup vs D" অংশে coverage-quality
      না হারিয়ে কে কতটা দ্রুত সেটা একনজরে
    - full_comparison.csv নামে একটা ফাইল এই ফোল্ডারে সেভ হবে (raw per-seed
      ডেটা, চাইলে নিজে আরও বিশ্লেষণ করতে পারবে)
"""
import sys
import a2e_relay_fixed_final_51 as relay

max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
num_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 50
seeds = list(range(1, num_seeds + 1))

print(f"Full comparison শুরু হচ্ছে: max_steps={max_steps}, seeds=1..{num_seeds}")
print("Policies: C (RelayScore), D (true FIS), D2 (surrogate FIS), T (TOPSIS), V (VIKOR)\n")

rows, summary = relay.run_topsis_vikor_comparison(
    seeds=seeds,
    max_steps=max_steps,
    policies=("C", "D", "D2", "T", "V"),
    out_csv="full_comparison.csv",
)

print("\n✅ শেষ! ফলাফল full_comparison.csv-এ সেভ হয়েছে।")
print("উপরের '--- Headline: selector time & speedup vs D ---' অংশটাই thesis-এর মূল টেবিলের জন্য সবচেয়ে গুরুত্বপূর্ণ।")
