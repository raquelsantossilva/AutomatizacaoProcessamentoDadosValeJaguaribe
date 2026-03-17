import psutil
ram = psutil.virtual_memory()
print(f"Total: {ram.total / 1e9:.1f} GB")
print(f"Disponível: {ram.available / 1e9:.1f} GB")
print(f"Uso: {ram.percent}%")