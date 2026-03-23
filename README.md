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
