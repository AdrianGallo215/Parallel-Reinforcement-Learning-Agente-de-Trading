"""
buy_and_hold_baseline.py — Retorno de comprar el primer dia evaluable y
mantener hasta el final, sobre la MISMA ventana de test que usa evaluate.py.

Sirve para verificar si el +16.63% del agente es simplemente Buy&Hold
disfrazado (si los dos numeros coinciden ~exactamente, lo es) o si el
agente aporto algo mas alla de comprar y no soltar.

Uso (desde la raiz del repo):
    py -m eval.buy_and_hold_baseline
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.data_loader import download_and_clean_data, get_train_test

TICKER = "AAPL"
START = "2020-01-01"
END = "2024-01-01"
WINDOW = 20
INITIAL_CASH = 10000.0

df = download_and_clean_data(TICKER, start=START, end=END)
_, test_data = get_train_test(df, test_ratio=0.2)

# Mismos indices que usa TradingEnv: min_step = window-1, max_step = len-2
min_step = WINDOW - 1
max_step = len(test_data) - 2

start_price = test_data["Close"].iloc[min_step]
end_price   = test_data["Close"].iloc[max_step]

bh_return_pct = (end_price / start_price - 1) * 100
bh_final_value = INITIAL_CASH * (end_price / start_price)

print("=" * 50)
print("  BASELINE: BUY & HOLD (mismo periodo de test)")
print("=" * 50)
print(f"  Precio inicial (paso {min_step}): ${start_price:,.2f}")
print(f"  Precio final   (paso {max_step}): ${end_price:,.2f}")
print(f"  Valor final:     ${bh_final_value:,.2f}")
print(f"  Retorno total:   {bh_return_pct:+.2f}%")
print("=" * 50)
print("\n  Compara este numero contra 'Total return' de eval.evaluate.")
print("  Si son casi iguales -> el agente aprendio Buy&Hold, no timing real.")