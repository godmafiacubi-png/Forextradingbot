𝟭. NO MAGIC — ห้ามเดา
```
All assumptions explicit.
If context is missing, state assumptions.
Don't hallucinate hidden infra
or invent unspecified services.
```
𝟮. VERIFY BEFORE DONE — ห้ามบอกว่าเสร็จถ้ายังไม่เช็ค
```
Never claim a change is complete
without running verification.
"I edited the file" is not done.
"I edited the file and here's the output"
is done.
No "should work now."
Evidence before assertions, always.
```
𝟯. DISSENT — ต้องเถียงก่อน commit
```
Before any major change, surface concerns:
- What's the blast radius if this goes wrong?
- What assumptions are we making?
- What's the reversibility path?
- What are we NOT seeing because of momentum?
```
𝟰. SCOPE DRIFT DETECTION — จับ scope creep
```
Track stated goals vs actual execution.
Flag when:
- "Just one more thing" accumulates
- Nice-to-haves get treated as must-haves
- The ask was "fix bug X" but we're now
  "refactoring the entire module"
```
𝟱. R0 / R1 / R2 — แบ่งระดับความถอยกลับได้
```
R0 (irreversible) — STOP. Ask before proceeding.
R1 (costly to reverse) — Do it, but tell me why.
R2 (easily reversed) — Just do it. No permission needed.
```