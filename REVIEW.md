# Full Program Review

วันที่ review: 2026-05-10

## Scope

ตรวจทั้งโปรเจกต์ในระดับ static/runtime-readiness โดยเน้น:

- entry points: `main.py`, `run_live.py`
- configuration: `config/settings.py`
- dashboard/monitoring modules
- local Python import graph
- syntax validity ของไฟล์ Python ทั้งหมด

## Findings

### 1. Blocking issue fixed: missing default dashboard module

`main.py` เลือก import dashboard ตาม `BOT_MODE`:

- `BOT_MODE == 'AGGRESSIVE'` ใช้ `monitoring.web_dashboard_aggressive`
- mode อื่นใช้ `monitoring.web_dashboard`

แต่ก่อนหน้านี้ repository ไม่มี `monitoring/web_dashboard.py` ทำให้ถ้าเปลี่ยน `BOT_MODE` เป็นค่าอื่น โปรแกรมจะ import fail ทันที แม้ logic ส่วนอื่นยังไม่เริ่มทำงาน

**Fix:** เพิ่ม `monitoring/web_dashboard.py` เป็น compatibility module ที่ re-export public API (`update_dashboard`, `add_log`, `start_dashboard`) จาก dashboard ที่มีอยู่ และตั้ง mode เป็น `DEFAULT` เพื่อให้ non-aggressive configuration import ได้

### 2. Dependency/runtime environment gap

สภาพแวดล้อมปัจจุบันยังไม่มี dependency หลักที่ระบุใน `requirements.txt` เช่น `MetaTrader5`, `pandas`, `numpy`, `tensorflow`, `torch`, `flask`, `dash` ดังนั้นยังไม่สามารถยืนยัน live runtime ด้วย `python main.py` ได้ใน container นี้

### 3. No automated tests present

รัน `pytest` แล้วไม่พบ test suite (`no tests ran`) ทำให้ coverage ของ behavior สำคัญ เช่น order execution, risk guard, signal generation และ dashboard integration ยังต้องพึ่ง manual/backtest validation

### 4. Risk areas for future hardening

- `main.py` และหลาย module ใช้ broad `except Exception` เยอะ ทำให้บาง failure อาจถูกกลืนและแสดงเป็น fallback behavior แทนที่จะ fail loudly
- มีการโหลด model ด้วย pickle/joblib ในหลายจุด ควรถือว่า trusted artifact เท่านั้น และไม่ควรโหลดไฟล์ model จาก source ที่ไม่เชื่อถือ
- `run_live.py` เป็น legacy entry point ตาม warning ในไฟล์ และขาด ML stack ใหม่ ควรใช้ `python main.py` เป็น entry point หลัก
- `README.md` ยังมีข้อมูลน้อยมาก ควรเพิ่ม setup/run instructions, required environment variables, และ MT5 prerequisites

## Verification performed

- `python -m compileall -q .` ผ่าน: syntax ของ Python files ทั้งหมด compile ได้
- custom AST parser ผ่าน: Python files ทั้งหมด parse ได้
- custom local import graph check ผ่านหลัง fix: ไม่พบ local module import ที่ missing
- `python -m pytest -q` รันได้แต่ไม่มี tests
- dependency import probe พบ missing dependencies ใน container ปัจจุบัน

## Recommended next steps

1. เพิ่ม smoke tests แบบไม่ต้องต่อ MT5 สำหรับ feature engineering, signal generation, risk sizing และ dashboard import
2. เพิ่ม `.env.example` เพื่อบอก required variables (`MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`, `TELEGRAM_TOKEN`, ฯลฯ)
3. ขยาย `README.md` ให้มี installation/run/backtest instructions
4. แยก dashboard สำหรับ conservative/default mode หากต้องการ UI ที่ต่างจาก aggressive mode จริง ๆ
5. ลด broad exception ใน critical paths โดย log traceback หรือ error context ที่เพียงพอสำหรับ live incident debugging
