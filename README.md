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




## 2. Nested Cross-Validation und Vergleich reduzierter Feature-Sets (`nested_cv_featureauswahl_hyperparameter_kalibrierung.ipynb`)

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




## 3. Finale Performance-, Kalibrations- und Metrikbewertung (`finale_performance_kalibration_metriken_plots.ipynb`)

Dieses Notebook vergleicht drei vordefinierte Datensätze bzw. Feature-Sets mit derselben **logistischen Regression** und bewertet nicht nur die Klassifikationsleistung, sondern auch die **Diskriminationsfähigkeit** und **Kalibrierung** der Modelle.

Verglichen werden:

- **Full**: alle numerischen Features
- **Top-38**: reduziertes Feature-Set mit 38 Variablen
- **Stable-70%**: kompaktes, stabileres Kern-Feature-Set

### Vorgehen

Die Daten werden eingelesen, die Zielvariable (`Verletzungsstatus`) getrennt und auf numerische Prädiktoren beschränkt. Für jedes Feature-Set wird dieselbe Modellpipeline verwendet. Die Bewertung erfolgt über eine **5×10 Repeated Stratified Cross-Validation** mit festen, zuvor optimierten Hyperparametern der logistischen Regression.

Berichtet werden zentrale Leistungsmetriken mit empirischen 95%-Konfidenzintervallen, darunter:

- Accuracy
- Balanced Accuracy
- F1 Score
- Recall
- Precision
- ROC-AUC
- PR-AUC
- Brier Score
- LogLoss

Zusätzlich werden für jedes Feature-Set eine **gepoolte ROC-Kurve** sowie ein **Kalibrationsplot** erstellt.

### Ziel

Dieses Notebook dient dazu,

1. die finale Modellleistung der verschiedenen Feature-Sets direkt zu vergleichen,
2. die Trennschärfe der Modelle zu beurteilen und
3. die Qualität der vorhergesagten Wahrscheinlichkeiten über Kalibrationsplots sichtbar zu machen.





## 4. Globale SHAP-Analyse des Stable-70%-Modells und Top-38_Modells (`shap_stable_70_percent.ipynb`)

Dieses Notebook dient der **globalen Modellinterpretation** der beiden finalen Modelle auf Basis von **SHAP-Werten**. Verwendet wird eine logistische Regression mit festen Hyperparametern sowie der **SHAP LinearExplainer**, sodass die Beiträge der einzelnen Merkmale in **Log-Odds-Einheiten** dargestellt werden.

### Vorgehen

Die Daten werden eingelesen, in Zielvariable und Prädiktoren aufgeteilt und über eine **Repeated Stratified Cross-Validation** ausgewertet. Kontinuierliche Variablen werden innerhalb jedes Trainingsfolds standardisiert, während die binäre Dummy-Variable `Geschlecht_weiblich` unskaliert bleibt. Für jeden Fold werden anschließend SHAP-Werte berechnet und über alle Testfolds zusammengeführt.

Auf dieser Basis werden zwei globale Kennwerte pro Feature bestimmt:

- **signed mean SHAP** zur Beschreibung der mittleren Wirkungsrichtung
- **mean(|SHAP|)** zur Beschreibung der globalen Relevanz

Zusätzlich enthält das Notebook mehrere Kontrollschritte, darunter:
- Prüfung auf `NaN`-/`Inf`-Werte in den SHAP-Ergebnissen
- Kontrolle der fold-spezifischen Expected Values
- Selbsttest zur numerischen Konsistenz der SHAP-Zerlegung
- vollständiges Feature-Ranking nach globaler SHAP-Bedeutung

### Visualisierung

Die globalen SHAP-Ergebnisse werden in mehreren Formen dargestellt:

- **Beeswarm-Plot** für die Verteilung und Richtung der SHAP-Werte
- **Balkenplot** zur geordneten Darstellung der globalen Feature-Bedeutung
- **Top-K-Visualisierung** der wichtigsten Merkmale für kompaktere Berichte und Präsentationen

### Ziel

Dieses Notebook dient dazu,

1. die wichtigsten Modelltreiber transparent zu machen,
2. Richtung und Stärke der einzelnen Feature-Beiträge global zu interpretieren und
3. die Ergebnisse in publikations- und präsentationstauglicher Form aufzubereiten.





## 5. Lokale SHAP-Analysen und Fallbeispiele (`shap_local.ipynb`)

Dieses Notebook dient der **lokalen Interpretation einzelner Modellentscheidungen** auf Basis von **SHAP-Werten**. Verwendet wird eine logistische Regression in Kombination mit dem **SHAP KernelExplainer**, sodass die Beiträge der Features direkt in **Wahrscheinlichkeitseinheiten** dargestellt werden können.

### Vorgehen

Die Daten werden eingelesen und über eine **Repeated Stratified Cross-Validation** ausgewertet. Kontinuierliche Variablen werden innerhalb jedes Trainingsfolds standardisiert, während die binäre Dummy-Variable `Geschlecht_weiblich` unskaliert bleibt. Für jeden Fold werden anschließend SHAP-Werte auf Basis der vorhergesagten Wahrscheinlichkeiten (`predict_proba`) berechnet und über alle Testfolds zusammengeführt.

Das Notebook enthält zusätzlich einen numerischen Selbsttest, bei dem geprüft wird, ob sich die Modellwahrscheinlichkeit für einzelne Beobachtungen korrekt aus **Expected Value + Summe der SHAP-Werte** rekonstruieren lässt.

### Visualisierung

Auf dieser Grundlage werden **lokale SHAP-Waterfall-Plots** für einzelne Beobachtungen erzeugt. Die Darstellung kombiniert:

- die **Basiswahrscheinlichkeit** des Modells
- die **individuelle Vorhersagewahrscheinlichkeit**
- die **beitragsstärksten Features**, die die Vorhersage in Richtung verletzt oder nicht verletzt verschieben

Für eine bessere Lesbarkeit wird der eigentliche Waterfall-Plot mit einem zusätzlichen Header zu einer gemeinsamen Grafik zusammengeführt.

### Ziel

Dieses Notebook dient dazu,

1. einzelne Modellentscheidungen auf Fallniveau nachvollziehbar zu machen,
2. die numerische Konsistenz der SHAP-Zerlegung zu überprüfen und
3. lokale Erklärungen in gut lesbarer Form für Berichte, Präsentationen oder exemplarische Fallanalysen bereitzustellen.
