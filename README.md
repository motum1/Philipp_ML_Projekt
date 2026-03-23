## 1. Ausreißerprüfung und Modellauswahl (`ausreißer_modellauswahl.ipynb`)

Dieses Notebook umfasst zwei frühe Schritte des ML-Workflows: die Prüfung potenzieller Ausreißer und einen ersten systematischen Vergleich klassischer Klassifikationsmodelle.

### Ausreißerprüfung

Im ersten Teil werden potenzielle Ausreißer in numerischen Excel-Spalten mit zwei komplementären Verfahren identifiziert:

- **Z-Score**
- **IQR-basierte Grenzwerte**

Auffällige Werte werden in eine kopierte Excel-Datei übernommen und farblich markiert:

- **Gelb** = Z-Score-Ausreißer
- **Orange** = IQR-Ausreißer
- **Rot** = von beiden Methoden erkannt

Dieser Schritt dient als praktische Datenqualitätskontrolle vor der weiteren Modellierung.

### Modellauswahl

Im zweiten Teil werden mehrere klassische Klassifikationsmodelle auf demselben Datensatz mit einer **5×10 Repeated Stratified Cross-Validation** verglichen. Alle Modelle werden in einem einheitlichen Preprocessing-Workflow bewertet. Berichtet werden unter anderem:

- Accuracy
- Balanced Accuracy
- F1 Score
- Recall
- Precision
- ROC-AUC
- PR-AUC
- Brier Score
- LogLoss

Die Ergebnisse werden als mittlere Leistung mit empirischen 95%-Konfidenzintervallen in einer übersichtlichen Vergleichstabelle zusammengefasst.

### Ziel

Dieses Notebook dient als erster Screening-Schritt, um

1. potenziell problematische Datenwerte sichtbar zu machen,
2. einen ersten Überblick über die Modellleistung verschiedener Klassifikatoren zu erhalten und
3. die Modellauswahl gezielt vorzubereiten.


## 2. Nested Cross-Validation und Vergleich reduzierter Feature-Sets

Dieser Abschnitt umfasst zwei zentrale Schritte der Modellverfeinerung: die methodisch saubere Entwicklung eines logistischen Regressionsmodells mit **Nested Cross-Validation** sowie den anschließenden Vergleich verschiedener **reduzierter Feature-Sets**.

### Nested Cross-Validation mit Feature-Selektion und Kalibrierung

Im ersten Teil wird ein logistisches Regressionsmodell in einem verschachtelten Validierungsansatz entwickelt. Innerhalb jedes Outer-Folds werden die Features zunächst über **Mutual Information** gerankt und anschließend mit einem **Korrelationsfilter** bereinigt, um redundante Variablen zu entfernen. Im inneren Cross-Validation-Schritt optimiert **Optuna** sowohl die Anzahl der verwendeten Features als auch die Regularisierungsstärke `C`.

Das finale Modell wird anschließend auf Basis der besten Parameter trainiert und zusätzlich mit **Platt Scaling** kalibriert, um die Vorhersagewahrscheinlichkeiten besser interpretierbar zu machen. Als Ergebnisse werden Leistungsmetriken mit empirischen Konfidenzintervallen, Hyperparameter-Zusammenfassungen, Feature-Häufigkeiten sowie gepoolte ROC- und Kalibrationsplots ausgegeben.

### Vergleich von Full-, Top-38- und Stable-70%-Datensätzen

Im zweiten Teil wird geprüft, wie sich die Modellleistung verändert, wenn statt aller numerischen Variablen nur reduzierte Feature-Sets verwendet werden. Verglichen werden:

- **Full**: alle numerischen Features
- **Top-38**: die 38 wichtigsten selektierten Merkmale
- **Stable-70%**: ein kompakteres Set aus besonders stabilen Features

Für alle drei Datensätze wird dieselbe logistische Regression mit identischem Preprocessing und **5×10 Repeated Stratified Cross-Validation** verwendet. Die Ergebnisse werden als Mittelwerte mit empirischen 95%-Konfidenzintervallen tabellarisch zusammengefasst.

### Ziel

Dieser Abschnitt dient dazu,

1. ein methodisch robustes logistisches Regressionsmodell ohne Data Leakage zu entwickeln,
2. stabile und wiederholt ausgewählte Merkmale zu identifizieren und
3. zu prüfen, ob reduzierte Feature-Sets eine vergleichbare Leistung wie das vollständige Modell erreichen.
