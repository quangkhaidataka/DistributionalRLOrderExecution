# HANDOFF.md — Tóm tắt context cuộc hội thoại (để tiếp tục ở chat mới)

> Cập nhật: 2026-07-18. Người dùng: Khai Nguyen — PhD Financial Mathematics, University of Manchester.
> Dự án: paper/thesis chapter **"Risk-Averse Optimal Execution via Distributional Reinforcement Learning"** (IQN + CVaR distortion), repo `~/Desktop/DisRL`, branch `fix/unify-arch-jd-recalibration`.

## 1. Mục tiêu & ràng buộc

- **Mục tiêu chính: nộp thesis PhD trong ~2 tháng** (deadline tự đặt từ ~18/7/2026). Chuẩn nhắm tới: pass viva mức **Aii (minor corrections)**. Topic 1 của thesis (giải một open problem, đã nộp Applied Mathematical Finance) là xương sống originality; paper này là Topic 2.
- Phê bình của supervisor đã được xử lý: (a) "jump model không hợp lý" → phản bác bằng literature (Moazeni et al. 2013 JCF; Cartea–Jaimungal; Lee–Mykland 2008; Christensen et al. 2014) + recalibrate jump như stress test; (b) "thiếu robustness" → đã chạy multi-seed, jump sensitivity, impact misspecification.
- Quyết định scope: **TAQ (AAPL 2014) giữ single-seed** (multi-seed chỉ trên simulation) — xử lý bằng limitation statement. Chỉ test trên AAPL, không multi-ticker (limitation).

## 2. Các audit đã hoàn thành (và đã sửa)

1. **Jump bug**: code cũ `poisson(λ·dt)` với dt=12 → 18 jumps/episode thay vì "0.3/period" như paper → JD env cũ thực chất là drift âm, không phải rare jumps. ĐÃ SỬA (per-period rate).
2. **Kiến trúc lệch 3 nơi**: JD run cũ dùng IQN 128/64 (43,398 params); AC/TAQ dùng 64/32 (11,462); DQN/DDQN 128 (18,566); paper ghi 64/64. ĐÃ THỐNG NHẤT: IQN 64/32 = **11,462 params**, DQN/DDQN width 64 = **5,190 params**, có param-count guard tự assert.
3. **Config chết**: yaml không được load; dataclass là source of truth; yaml đã sửa thành reference-only khớp code.
4. **Paper–code mismatch** (PAPER_FIXES.md): cos n 64→32, target update C 100→500, TAQ episodes=50k/q0=5k/η=1e-5/a=1e-4 (đã đúng), câu "same architecture" phải viết lại (nêu 11,462 vs 5,190, cite Dabney 2018), notation collision λ/η/γ, abstract percentages phải tính lại.
5. **N7**: dữ liệu AAPL_2014.parquet chưa split-adjust (7:1 tháng 6/2014) — đã verify KHÔNG hỏng kết quả test (arrival price per-episode, test window Oct–Dec post-split); ghi limitation, không sửa.

## 3. Calibration JD cuối cùng (LOCKED — không dò thêm)

- **λ_J = 0.05/period, σ_J = $0.16, μ_J = 0 (symmetric), selection = min val-CVaR₉₅** (cvar-select).
- Quá trình: scan 6 ô pre-registered → (0.05, 0.16) thắng; ma trận 2×2 {σ}×{selection rule} xác nhận; fallback (0.05, 0.12) đã khai báo trước nhưng không dùng; fallback rule được **amend có ghi chép** (DDQN collapse tái phân loại thành finding, không phải env failure).
- Old degenerate runs lưu tại `results/_archive_jd_stress_seed42/` (có thể dùng làm "extreme stress" appendix).

## 4. Kết quả hiện có (tất cả trên CPU — phát hiện CPU nhanh hơn MPS ~6× với mạng nhỏ)

- **AC (5 seeds)**: mọi agent tương đương (~1.9–2.2 CVaR₉₅); distortion trơ dưới Gaussian — đúng dự đoán lý thuyết, giữ làm sanity check.
- **JD (5 seeds, σ=0.16)**: TWAP CVaR₉₅ 12.26±0.35 · DQN 3.68±0.26 · DDQN 2.085±0.001 (**dump collapse 5/5 seeds**) · IQN-neutral 3.71±1.47 · IQN-CVaR₀.₉₅ 3.61±1.30. RL cắt ~**73% tail vs TWAP** (claim xương sống). Δ differentiation (neutral−CVaR₀.₉₅) = **+0.098 ± 0.226** (đúng chiều, nhỏ, không seed-robust). **IQN seed-variance ±1.47 cao gấp ~6× DQN** — điểm yếu cần tự nêu trong Discussion.
- **TAQ seed-42 (single-seed by design)**: TWAP CVaR 54.89/Max 207.8 · DQN(64) 19.67/Max **147.1** · DDQN(64) 6.38/38.0 · IQN-CVaR 6.13/**6.24** (giữ checkpoint cũ). Headline: −88.8% CVaR vs TWAP (stable), Max thấp hơn DQN ~23.6×. **Đây là nơi distributional advantage rõ nhất.**
- **Jump sensitivity** (λ ∈ {0.025, 0.05, 0.10}, σ=0.16): ba chế độ — low: IQN under-hedge (11.8, tệ hơn TWAP 8.2); base: sweet spot (3.46, −73%); high: IQN tự dump (2.084 — hợp lý vì dump là tối ưu thật khi λ cao). TWAP tail đơn điệu 8.2→12.6→16.7.
- **Impact misspec 3×3** (η,γ × {0.5,1,2}): mượt, đơn điệu, không instability.
- **α-sweep**: AC phẳng tuyệt đối (đúng lý thuyết); TAQ hiệu ứng ngưỡng ở α≈1 (Max 40.8→6.24); JD đúng chiều nhẹ (3.54→3.35 tại α=0.5).

## 5. Narrative đã thống nhất cho paper (cấu trúc 3 môi trường)

- **AC** = sanity: mọi phương pháp tương đương, distortion trơ khi không có tail (evidence phương pháp không thừa).
- **JD** = RL ≫ static (−73%); IQN non-degenerate vs **DDQN corner collapse 5/5** (cơ chế 4 mắt xích: fat-tail reward → scalar Q dao động → policy chớp nháy qua dump → min-val-CVaR selection vớt đúng snapshot dump; IQN chặn từ mắt xích 1 nhờ học phân phối); dial đúng chiều nhưng khiêm tốn (nêu α=0.5); sensitivity 3 chế độ.
- **TAQ** = nơi dial cắn thật và IQN-CVaR áp đảo mọi risk metric.
- **Dump corner là feature không phải bug**: trong stress scenario không ràng buộc, thanh lý tức thời là CVaR-optimal thật — báo cáo trung thực qua baseline **Immediate Liquidation (IL)**; participation constraint = future work. Claim L669 cũ ("IQN ↓10.7% CVaR vs DQN") **bỏ** — DQN ≈ IQN trên JD.
- Selection rule: **GIỮ min val-CVaR cho bảng chính** (DDQN=dump hiển thị trung thực cạnh hàng IL); appendix hai-luật (cvar vs mean) làm robustness; KHÔNG dùng luật "min CVaR loại dump" (= handicap baseline, nguy hiểm).

## 6. Quy trình làm việc đã thiết lập

- **Claude Code**: code + verify + jobs ngắn; **harness reap ~34 phút** → mọi việc dài phải staged thành jobs <30 phút (pattern đã proven: per-cell, per-agent với `--only-agent`/`--assemble-eval`, equivalence check bắt buộc) hoặc chạy ở **terminal của user** (`caffeinate -dims ... | tee log`).
- Living docs trong repo: `PLAN.md`, `DECISION.md`, `PROGRESS.md`, `RUNBOOK.md`, `PAPER_FIXES.md`, `VERIFY.md`, `results/_batch_bc_report.md`. **`main_paper.tex` CHƯA từng bị sửa.**
- Mọi run dump config JSON; không bao giờ ghi đè `results/*` (chỉ archive); param guards 11,462/5,190 in mọi lần build.

## 7. ĐANG Ở BƯỚC NÀY (trạng thái khi kết thúc chat)

- Batches A, B1 (sim multi-seed), C (sensitivity + misspec) **hoàn thành**. Batch D (width ablation) **bỏ**. TAQ multi-seed **bỏ by design**.
- **Prompt "Batch E" đã được đưa cho user, chưa có kết quả** — gồm: **E1** IL baseline (eval-only, 3 settings); **E2** thêm hàng DDQN vào jump sensitivity (3 λ levels); **E3** selection-rule appendix: re-validate toàn bộ checkpoints (val=1,200 eps, CRN) → re-select 2 luật (cvar/mean) → bảng song song + kiểm chứng dự đoán "DDQN un-dumps under mean-selection".

## 8. Bước tiếp theo (sau Batch E)

1. User paste Batch E report → review (đặc biệt: IL row JD ≈ 2.084 sanity; DDQN sensitivity có dump ở mọi level không; dự đoán mean-selection có đúng không).
2. **Paper surgery prompt** (chặng cuối): lần đầu cho phép sửa `main_paper.tex`, trên branch riêng, mỗi commit một nhóm theo PAPER_FIXES: (i) số bảng mới + jump params + C=500 + n=32; (ii) percentages abstract tính lại (88.8%/97.0% giữ; 5.2×/7.3× → số mới ~23.6×/6.1×; bỏ L669); (iii) đoạn fair-comparison (11,462 vs 5,190); (iv) reframe JD stress-test + Poisson per-period form; (v) thêm IL baseline + appendices (multi-seed, sensitivity, misspec, selection-rule); (vi) Discussion mới (cơ chế DDQN, dump corner, Δ nhỏ + giải thích action space rời rạc, IQN seed-variance); (vii) Limitations (TAQ single-seed/single-ticker, N7 split-adjust, validation 300 eps nhiễu); (viii) notation λ_J/η/γ disambiguation; xóa abstract chết L76–80.
3. User review diff từng phần → đưa supervisor đọc → hoàn thiện thesis.

## 9. Các con số cần nhớ nhanh

| Item | Giá trị |
|---|---|
| IQN params / DQN,DDQN params | 11,462 / 5,190 (state 5, actions 6) |
| Dump cost JD (η·q₀/Δt) | 2.083 bps — chữ ký nhận diện dump |
| JD locked | λ=0.05, σ_J=$0.16, μ_J=0, cvar-select, ckpt tốt: IQN ep7000 (seed42) |
| TAQ setup | q0=5,000; η=1e-5; a=1e-4; 50k episodes; fold Jan–Jul/Aug–Sep/Oct–Dec 2014; IQN checkpoint GIỮ NGUYÊN |
| Sim setup | N=5, T=60′, q0=100k, η=2.5e-6, γ=2.5e-7, σ=0.00095, 30k episodes, C=500 |
