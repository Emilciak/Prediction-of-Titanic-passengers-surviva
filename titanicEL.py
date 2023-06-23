from pyspark.sql import SparkSession
from pyspark.sql.functions import avg
from pyspark.sql.functions import sum
from pyspark.sql.functions import count
from pyspark.sql.functions import max, min
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Tworzenie sesji Spark
spark = SparkSession.builder \
    .appName("Titanic Data Analysis") \
    .getOrCreate()

# Wczytywanie danych z pliku CSV
data = spark.read.csv("titanic2.csv", header=True, inferSchema=True)

# Dane testowe
test = data.sample(fraction=0.2, seed=42)

# Wyświetlanie kilku wierszy danych
data.show()

# Wyświetlanie kilku wierszy danych testowych
test.show()


################################dane testowe #################################

# Obliczanie liczby kobiet i mężczyzn
print("Obliczanie liczby kobiet i mężczyzn w zbiorze testowym")
gender_count = test.groupBy("Sex").agg(count("PassengerId").alias("Count"))

# Wyświetlanie wyników
gender_count.show()

# Grupowanie danych i obliczanie średniej wieku w każdej klasie
print(" Grupowanie danych i obliczanie średniej wieku w każdej klasie w zbiorze testowym")
avg_age_by_class = test.groupBy("Pclass").agg(avg("Age").alias("AvgAge"))

# Wyświetlanie wyników
avg_age_by_class.show()

# Obliczanie średniej wieku pasażerów
print(" Obliczanie średniej wieku pasażerów w zbiorze testowym")
average_age = test.select(avg("Age")).first()[0]

print("")
# Wyświetlanie wyniku
print("Średni wiek pasażerów: {:.2f}".format(average_age))

# Obliczanie liczby pasażerów w zależności od przeżycia
print(" Obliczanie liczby pasażerów w zależności od przeżycia zbiorze testowym")
passenger_count_by_survived = test.groupBy("Survived").agg(count("PassengerId").alias("Count"))

# Wyświetlanie wyników
passenger_count_by_survived.show()

# Obliczanie najwyższej i najniższej opłaty za bilet
print(" Obliczanie najwyższej i najniższej opłaty za bilet w zbiorze testowym")
max_fare = test.select(max("Fare")).first()[0]
min_fare = test.select(min("Fare")).first()[0]
print("")
# Wyświetlanie wyników
print("Najwyższa opłata za bilet: {:.2f}".format(max_fare))
print("Najniższa opłata za bilet: {:.2f}".format(min_fare))
print("")
# Obliczanie średniej opłaty za bilet w zależności od klasy
print(" Obliczanie średniej opłaty za bilet w zależności od klasy w zbiorze testowym")
avg_fare_by_class = test.groupBy("Pclass").agg(avg("Fare").alias("AvgFare"))

# Wyświetlanie wyników
avg_fare_by_class.show()

# Obliczanie liczby pasażerów w zależności od liczby rodzeństwa/małżonków
print(" Obliczanie liczby pasażerów w zależności od liczby rodzeństwa/małżonków w zbiorze testowym")
passenger_count_by_sibsp = test.groupBy("SibSp").agg(count("PassengerId").alias("Count"))

# Wyświetlanie wyników
passenger_count_by_sibsp.show()


# Obliczanie liczby pasażerów w zależności od portu wypłynięcia
print(" Obliczanie liczby pasażerów w zależności od portu wypłynięcia w zbiorze testowym")
passenger_count_by_embarked = test.groupBy("Embarked").agg(count("PassengerId").alias("Count"))

# Wyświetlanie wyników
passenger_count_by_embarked.show()

# Obliczanie liczby pasażerów w poszczególnych klasach
print(" Obliczanie liczby pasażerów w poszczególnych klasach w zbiorze testowym")
passenger_count_by_class = test.groupBy("Pclass").agg(count("PassengerId").alias("Count"))

# Wyświetlanie wyników
passenger_count_by_class.show()

# Obliczanie liczby pasażerów w zależności od kabiny
print("Obliczanie liczby pasażerów w zależności od kabiny w zbiorze testowym")
passenger_count_by_cabin = test.groupBy("Cabin").agg(count("PassengerId").alias("Count"))

# Wyświetlanie wyników
passenger_count_by_cabin.show()

# Obliczanie liczby pasażerów w zależności od płci i klasy
print("Obliczanie liczby pasażerów w zależności od płci i klasy w zbiorze testowym")
passenger_count_by_gender_class = test.groupBy("Sex", "Pclass").agg(count("PassengerId").alias("Count"))

# Wyświetlanie wyników
passenger_count_by_gender_class.show()




################################################################
# Obliczanie liczby kobiet i mężczyzn
gender_count = data.groupBy("Sex").agg(count("PassengerId").alias("Count"))

# Wyświetlanie wyników
gender_count.show()

# Grupowanie danych i obliczanie średniej wieku w każdej klasie
avg_age_by_class = data.groupBy("Pclass").agg(avg("Age").alias("AvgAge"))

# Wyświetlanie wyników
avg_age_by_class.show()

# Obliczanie średniej wieku pasażerów
average_age = data.select(avg("Age")).first()[0]

# Wyświetlanie wyniku
print("Średni wiek pasażerów: {:.2f}".format(average_age))

# Obliczanie liczby pasażerów w zależności od przeżycia
passenger_count_by_survived = data.groupBy("Survived").agg(count("PassengerId").alias("Count"))

# Wyświetlanie wyników
passenger_count_by_survived.show()

# Obliczanie najwyższej i najniższej opłaty za bilet
max_fare = data.select(max("Fare")).first()[0]
min_fare = data.select(min("Fare")).first()[0]

# Wyświetlanie wyników
print("Najwyższa opłata za bilet: {:.2f}".format(max_fare))
print("Najniższa opłata za bilet: {:.2f}".format(min_fare))

# Obliczanie średniej opłaty za bilet w zależności od klasy
avg_fare_by_class = data.groupBy("Pclass").agg(avg("Fare").alias("AvgFare"))

# Wyświetlanie wyników
avg_fare_by_class.show()

# Obliczanie liczby pasażerów w zależności od liczby rodzeństwa/małżonków
passenger_count_by_sibsp = data.groupBy("SibSp").agg(count("PassengerId").alias("Count"))

# Wyświetlanie wyników
passenger_count_by_sibsp.show()


# Obliczanie liczby pasażerów w zależności od portu wypłynięcia
passenger_count_by_embarked = data.groupBy("Embarked").agg(count("PassengerId").alias("Count"))

# Wyświetlanie wyników
passenger_count_by_embarked.show()

# Obliczanie liczby pasażerów w poszczególnych klasach
passenger_count_by_class = data.groupBy("Pclass").agg(count("PassengerId").alias("Count"))

# Wyświetlanie wyników
passenger_count_by_class.show()

# Obliczanie liczby pasażerów w zależności od kabiny
passenger_count_by_cabin = data.groupBy("Cabin").agg(count("PassengerId").alias("Count"))

# Wyświetlanie wyników
passenger_count_by_cabin.show()

# Obliczanie liczby pasażerów w zależności od płci i klasy
passenger_count_by_gender_class = data.groupBy("Sex", "Pclass").agg(count("PassengerId").alias("Count"))

# Wyświetlanie wyników
passenger_count_by_gender_class.show()
################################AGLORYTM DRZEWA DECYZYJNEGO NA ZBIORZE TESTOWYM #################################
print("AGLORYTM DRZEWA DECYZYJNEGO NA ZBIORZE TESTOWYM ")
# Przygotowanie danych - usunięcie wierszy zawierających brakujące dane. na.drop() usuwa wiersze zawierające wartości null.
test = test.na.drop()

# Indeksowanie kolumn kategorycznych - kolumny "Sex" i "Embarked" są kategoryczne, co oznacza, że zawierają wartości tekstowe. StringIndexer zamienia te wartości tekstowe na liczbowe indeksy. Parametr inputCols określa nazwy kolumn wejściowych, a outputCols określa nazwy kolumn wyjściowych z indeksami. Wynik jest przechowywany w ramce danych indexed_data.
indexer = StringIndexer(inputCols=["Sex", "Embarked"], outputCols=["SexIndex", "EmbarkedIndex"])
indexed_data = indexer.fit(test).transform(test)

# Tworzenie wektora cech - VectorAssembler łączy wybrane kolumny cech w pojedynczą kolumnę wektorową. Parametr inputCols to lista nazw kolumn wejściowych, a outputCol to nazwa kolumny wyjściowej, w której zostanie umieszczony wektor cech. handleInvalid="keep" oznacza zachowanie wartości niezgodnych z modelem, które są reprezentowane jako NaN lub null. Wynik jest przechowywany w ramce danych assembled_data.
#assembler = VectorAssembler(inputCols=["Pclass", "SexIndex"], outputCol="features", handleInvalid="keep")
assembler = VectorAssembler(inputCols=["Pclass", "SexIndex", "Age", "SibSp", "Parch", "Fare", "EmbarkedIndex"], outputCol="features", handleInvalid="keep")
assembled_data = assembler.transform(indexed_data)

# Podział danych na zbiór treningowy i testowy - randomSplit dzieli ramkę danych assembled_data na zbiory treningowy i testowy w proporcji 0.8:0.2. Zbiór treningowy jest przechowywany w training_data, a zbiór testowy w test_data.
(training_data, test_data) = assembled_data.randomSplit([0.8, 0.2])

# Inicjalizacja modelu i trenowanie drzewa decyzyjnego - DecisionTreeClassifier jest inicjalizowany z parametrem labelCol ustawionym na "Survived", co oznacza, że to kolumna "Survived" jest etykietą, której model będzie się uczył. Parametr featuresCol wskazuje na kolumnę zawierającą wektor cech. Następnie model jest trenowany na zbiorze treningowym za pomocą fit.
dt = DecisionTreeClassifier(labelCol="Survived", featuresCol="features")
model = dt.fit(training_data)

#Wykonanie predykcji na zbiorze testowym - transform stosuje na zbiorze testowym nauczone drzewo decyzyjne i zwraca ramkę danych z dodaną kolumną "prediction", która zawiera przewidywane etykiety.
predictions = model.transform(test_data)

# Ewaluacja modelu - MulticlassClassificationEvaluator jest używany do obliczenia różnych metryk oceny klasyfikacji wieloklasowej. Parametr labelCol wskazuje na kolumnę z prawdziwymi etykietami, a predictionCol na kolumnę z przewidywanymi etykietami. Metryki, takie jak "accuracy", "weightedPrecision", "weightedRecall" i "f1", są obliczane za pomocą evaluate.
evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedPrecision")
precision = evaluator.evaluate(predictions)

evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedRecall")
recall = evaluator.evaluate(predictions)

evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)

# # Obliczenie obszaru pod krzywą ROC - MulticlassClassificationEvaluator oblicza obszar pod krzywą ROC na podstawie etykiet i przewidywanych wartości. Parametr labelCol wskazuje na kolumnę z prawdziwymi etykietami.
evaluator = MulticlassClassificationEvaluator(labelCol="Survived")
area_under_roc = evaluator.evaluate(predictions)


## Wybieramy potrzebne kolumny z ramki danych predictions
predictions_selected = predictions.select("Survived", "prediction")

# Wyświetlamy tabelę z rzeczywistymi i przewidzianymi wartościami
predictions_selected.show()

# Wyświetlanie wyników
print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1_score))
print("Area Under ROC: {:.2f}".format(area_under_roc))


################################ALGORYMT DRZEWA DECYZYJNEGO ################################
print("ALGORYMT DRZEWA DECYZYJNEGO")
# Przygotowanie danych - usunięcie wierszy zawierających brakujące dane. na.drop() usuwa wiersze zawierające wartości null.
data = data.na.drop()

# Indeksowanie kolumn kategorycznych - kolumny "Sex" i "Embarked" są kategoryczne, co oznacza, że zawierają wartości tekstowe. StringIndexer zamienia te wartości tekstowe na liczbowe indeksy. Parametr inputCols określa nazwy kolumn wejściowych, a outputCols określa nazwy kolumn wyjściowych z indeksami. Wynik jest przechowywany w ramce danych indexed_data.
indexer = StringIndexer(inputCols=["Sex", "Embarked"], outputCols=["SexIndex", "EmbarkedIndex"])
indexed_data = indexer.fit(data).transform(data)

# Tworzenie wektora cech - VectorAssembler łączy wybrane kolumny cech w pojedynczą kolumnę wektorową. Parametr inputCols to lista nazw kolumn wejściowych, a outputCol to nazwa kolumny wyjściowej, w której zostanie umieszczony wektor cech. handleInvalid="keep" oznacza zachowanie wartości niezgodnych z modelem, które są reprezentowane jako NaN lub null. Wynik jest przechowywany w ramce danych assembled_data.
#assembler = VectorAssembler(inputCols=["Pclass", "SexIndex"], outputCol="features", handleInvalid="keep")
assembler = VectorAssembler(inputCols=["Pclass", "SexIndex"], outputCol="features", handleInvalid="keep")
assembled_data = assembler.transform(indexed_data)

# Podział danych na zbiór treningowy i testowy - randomSplit dzieli ramkę danych assembled_data na zbiory treningowy i testowy w proporcji 0.8:0.2. Zbiór treningowy jest przechowywany w training_data, a zbiór testowy w test_data.
(training_data, test_data) = assembled_data.randomSplit([0.8, 0.2])

# Inicjalizacja modelu i trenowanie drzewa decyzyjnego - DecisionTreeClassifier jest inicjalizowany z parametrem labelCol ustawionym na "Survived", co oznacza, że to kolumna "Survived" jest etykietą, której model będzie się uczył. Parametr featuresCol wskazuje na kolumnę zawierającą wektor cech. Następnie model jest trenowany na zbiorze treningowym za pomocą fit.
dt = DecisionTreeClassifier(labelCol="Survived", featuresCol="features")
model = dt.fit(training_data)

#Wykonanie predykcji na zbiorze testowym - transform stosuje na zbiorze testowym nauczone drzewo decyzyjne i zwraca ramkę danych z dodaną kolumną "prediction", która zawiera przewidywane etykiety.
predictions = model.transform(test_data)

# Ewaluacja modelu - MulticlassClassificationEvaluator jest używany do obliczenia różnych metryk oceny klasyfikacji wieloklasowej. Parametr labelCol wskazuje na kolumnę z prawdziwymi etykietami, a predictionCol na kolumnę z przewidywanymi etykietami. Metryki, takie jak "accuracy", "weightedPrecision", "weightedRecall" i "f1", są obliczane za pomocą evaluate.
evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedPrecision")
precision = evaluator.evaluate(predictions)

evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedRecall")
recall = evaluator.evaluate(predictions)

evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)

# Obliczenie obszaru pod krzywą ROC - MulticlassClassificationEvaluator oblicza obszar pod krzywą ROC na podstawie etykiet i przewidywanych wartości. Parametr labelCol wskazuje na kolumnę z prawdziwymi etykietami.
evaluator = MulticlassClassificationEvaluator(labelCol="Survived")
area_under_roc = evaluator.evaluate(predictions)


## Wybieramy potrzebne kolumny z ramki danych predictions
predictions_selected = predictions.select("Survived", "prediction")

# Wyświetlamy tabelę z rzeczywistymi i przewidzianymi wartościami
predictions_selected.show()

# Wyświetlanie wyników
print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1_score))
print("Area Under ROC: {:.2f}".format(area_under_roc))