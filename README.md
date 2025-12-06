# Accelerometer-Based Motion Detection

## Projekt: Detekcja Ruchu z Akcelerometru z Kompresją Modelu

### Autorzy: Filip Izworski, Jakub Kozdrój, Mikołaj Kołodziej

### Przegląd Projektu
Kompletny pipeline do rozpoznawania aktywności ludzkiej (Human Activity Recognition - HAR) wykorzystujący dane z akcelerometru i żyroskopu smartfona. Projekt implementuje techniki kompresji modeli neuronowych (pruning i kwantyzacja) dla wdrożeń na urządzeniach brzegowych (edge devices).

### Cel Projektu
Demonstracja efektywnej kompresji modeli głębokiego uczenia, gdzie ograniczona pamięć i moc obliczeniowa wymagają małych i wydajnych modeli.

## Funkcjonalności

### Przetwarzanie Danych
- Automatyczne pobieranie i ekstrakcja **UCI HAR Dataset**
- Ładowanie 6-kanałowych sygnałów (3 akcelerometry + 3 żyroskopy)
- Normalizacja per-kanałowa z użyciem StandardScaler
- Przygotowanie DataLoaderów PyTorch do treningu i testowania

### Architektura Modelu
- **Lekka 1D-CNN** (~13,000 parametrów)
- 2 warstwy konwolucyjne (32 i 64 filtry)
- Global Average Pooling + warstwa fully connected
- 6 klas wyjściowych (aktywności):
  - 1: CHODZENIE
  - 2: CHODZENIE PO SCHODACH W GÓRĘ
  - 3: CHODZENIE PO SCHODACH W DÓŁ
  - 4: SIEDZENIE
  - 5: STANIE
  - 6: LEŻENIE

### Techniki Kompresji

#### 1. Pruning
- **Magnitude-based pruning** - usuwanie wag o najmniejszej wartości bezwzględnej
- Możliwość regulacji stopnia pruningu (domyślnie 30%)
- Zapis w **formacie rzadkim (sparse)** - tylko niezerowe wartości
- Faktyczna redukcja rozmiaru pliku

#### 2. Kwantyzacja
- Symetryczna kwantyzacja do niższej precyzji:
  - **INT8** - 8-bitowa (docelowo 25% rozmiaru FP32)
  - **INT4** - 4-bitowa (docelowo 12.5% rozmiaru FP32)
  - **INT16** - 16-bitowa (dla porównania)
- Zachowanie skal kwantyzacji dla dekwantyzacji

#### 3. Kombinacja Technik
- Sekwencyjne zastosowanie pruningu i kwantyzacji
- Maksymalna kompresja przy minimalnej utracie dokładności
- Format sparse+quantized dla optymalnego przechowywania

### Ewaluacja i Wizualizacja
- Porównanie dokładności i rozmiaru względem baseline'u FP32
- Generowanie wykresów porównawczych
- Obliczanie procentowej redukcji rozmiaru
- Analiza trade-off: dokładność vs rozmiar

## Instalacja

### Wymagania Wstępne
- Python 3.8 lub nowszy

### Instalacja Zależności

```bash
# Sklonuj repozytorium
git clone https://github.com/filizw/accelerometer-motion-detection.git
cd accelerometer-motion-detection

# Zainstaluj wymagane pakiety
pip install -r requirements.txt
```

## Użycie

```bash
# Tylko trening modelu baseline FP32
python har_pipeline.py

# Pruning 30% (domyślnie)
python har_pipeline.py --prune

# Pruning z customowym stopniem (np. 50%)
python har_pipeline.py --prune --prune_amount 0.5

# Kwantyzacja INT8 (domyślnie)
python har_pipeline.py --quantize

# Kwantyzacja INT4
python har_pipeline.py --quantize --quantize_bits 4

# Obie techniki: pruning 30% + kwantyzacja INT8
python har_pipeline.py --prune --quantize
```

## Struktura Projektu

```bash
accelerometer-motion-detection/
├── har_pipeline.py             # Główny skrypt pipeline
├── requirements.txt            # Zależności Pythona
├── README.md                   # Ta dokumentacja
├── .gitignore                  # Ignorowane pliki Gita
│
├── results/                    # Wygenerowane wyniki
│   ├── model_fp32.pt                   # Model baseline FP32
│   ├── model_pruned_30_sparse.npz      # Przykład modelu po pruningu
│   ├── model_quantized_8bit.npz        # Przykład modelu skwantyzowanego
│   ├── compression_results.png         # Wykres porównawczy
│   └── compression_summary.txt         # Podsumowanie wyników
│
└── UCI_HAR_Dataset/            # Dataset (pobierany automatycznie)
```

## Wyniki

**Model bazowy FP32:** Rozmiar = 48.7 KB, Dokładność = 0.8205

| Technika                | Rozmiar względem FP32 | Dokładność względem FP32 | Kluczowa obserwacja                    |
|-------------------------|----------------------|-------------------------|----------------------------------------|
| FP32 Baseline           | 1.00x               | 1.000x                  | Model referencyjny                     |
| Pruning 30%             | 0.752x (75.2%)      | 1.009x (100.9%)         | Mniejszy i dokładniejszy               |
| Kwantyzacja INT8        | 0.230x (23%)        | 0.995x (99.5%)          | 4.3× kompresja, minimalna utrata       |
| Pruning 30% + INT8      | 0.303x (30.3%)      | 1.009x (100.9%)         | Optymalny kompromis: 3.3× bez strat    |                     |