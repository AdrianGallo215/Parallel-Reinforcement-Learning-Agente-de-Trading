"""
diagnose_state_dependence.py — diagnóstico BARATO (no reentrena).

Carga un checkpoint ya guardado y mide, sobre TODO el test set, la
DESVIACIÓN ESTÁNDAR de las probabilidades de acción a lo largo de los pasos.

Interpretación:
  - std ~ 0 en las 3 acciones  -> la red produce casi la MISMA distribución
    sin importar el estado que observa: nunca aprendió a condicionar su
    salida en el input (política colapsada / degenerada).
  - std claramente > 0          -> sí hay dependencia del estado; el problema
    sería de cantidad de entrenamiento, no de "no aprende nada".

Uso:
    python eval/diagnose_state_dependence.py --checkpoint checkpoints/tune_default.pt
    python eval/diagnose_state_dependence.py --checkpoint checkpoints/tune_variant_1.pt
"""
import sys, os, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.data_loader import download_and_clean_data, get_train_test
from env.trading_env import TradingEnv
from model.actor_critic import ActorCritic


def diagnose(model, test_data, window=20):
    env = TradingEnv(data=test_data, window=window)

    # Recorremos TODOS los pasos evaluables y guardamos la observación real.
    # Nota: usamos position=0 fijo para aislar la dependencia en los features
    # de mercado (si quieres, corre también con position=1 para comparar).
    obses = []
    for step in range(env.min_step, env.max_step + 1):
        env.current_step = step
        env.position = 0
        obses.append(env._get_obs())
    obs_t = torch.FloatTensor(np.array(obses))

    model.eval()
    with torch.no_grad():
        probs, values = model(obs_t)
    probs = probs.numpy()          # (n_steps, 3)
    values = values.numpy().ravel()

    names = ["Hold", "Buy", "Sell"]
    print(f"\n{'='*56}")
    print(f"  DIAGNÓSTICO DE DEPENDENCIA DEL ESTADO")
    print(f"  pasos evaluados: {len(obses)}")
    print(f"{'='*56}")
    for i, n in enumerate(names):
        print(f"  {n:4s}  mean={probs[:,i].mean():.4f}   "
              f"std={probs[:,i].std():.4f}   "
              f"min={probs[:,i].min():.4f}  max={probs[:,i].max():.4f}")
    print(f"  ----")
    print(f"  V(s)  mean={values.mean():.4f}   std={values.std():.4f}   "
          f"(un critic que aprendió también debería VARIAR con el estado)")

    max_std = probs.std(axis=0).max()
    print(f"\n  std máximo entre acciones = {max_std:.4f}")
    if max_std < 0.01:
        print("  => VEREDICTO: salida casi constante. La red NO condiciona en el")
        print("     estado (política colapsada). No es cuestión de más episodios.")
    elif max_std < 0.05:
        print("  => VEREDICTO: dependencia del estado MUY débil. Señal marginal.")
    else:
        print("  => VEREDICTO: sí hay dependencia del estado apreciable.")
    print(f"{'='*56}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2024-01-01")
    args = ap.parse_args()

    df = download_and_clean_data(args.ticker, start=args.start, end=args.end)
    _, test_data = get_train_test(df, test_ratio=0.2)
    model = ActorCritic.load_checkpoint(args.checkpoint)
    print(f"Loaded: {args.checkpoint}  |  test rows: {len(test_data)}")
    diagnose(model, test_data)