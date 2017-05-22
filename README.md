# java-outliers
Package provide java implementation of outlier detection using normal distribution for multi-variate datasets

# Install

Add the following dependency to your POM file:

```xml
<dependency>
  <groupId>com.github.chen0040</groupId>
  <artifactId>java-outliers</artifactId>
  <version>1.0.1</version>
</dependency>
```


# Usage

The anomaly detection algorithms takes data that is prepared and stored in a data frame (Please refers to this [link](https://github.com/chen0040/java-data-frame) on how to create a data frame from file or from scratch)

The MultiVariateNormalOutliers can be trained using unsupervised learning. 

To create and train the MultiVariateNormalOutliers, run the following code:

```java
MultiVariateNormalOutliers method = new MultiVariateNormalOutliers();
method.fit(trainingData);
```
 
To test the trained method on new data, run:

```java
boolean outlier = method.isAnomaly(dataRow);
```

### Complete sample code for LOF

The problem that we will be using as demo as the following anomaly detection problem:

![scki-learn example for one-class](http://scikit-learn.org/stable/_images/sphx_glr_plot_oneclass_001.png)

Below is the sample code which illustrates how to use MultiVariateNormalOutliers to detect outliers in the above problem:

```java

DataQuery.DataFrameQueryBuilder schema = DataQuery.blank()
      .newInput("c1")
      .newInput("c2")
      .newOutput("anomaly")
      .end();

Sampler.DataSampleBuilder negativeSampler = new Sampler()
      .forColumn("c1").generate((name, index) -> randn() * 0.3 + (index % 2 == 0 ? -2 : 2))
      .forColumn("c2").generate((name, index) -> randn() * 0.3 + (index % 2 == 0 ? -2 : 2))
      .forColumn("anomaly").generate((name, index) -> 0.0)
      .end();

Sampler.DataSampleBuilder positiveSampler = new Sampler()
      .forColumn("c1").generate((name, index) -> rand(-4, 4))
      .forColumn("c2").generate((name, index) -> rand(-4, 4))
      .forColumn("anomaly").generate((name, index) -> 1.0)
      .end();

DataFrame trainingData = schema.build();

trainingData = negativeSampler.sample(trainingData, 200);
trainingData = positiveSampler.sample(trainingData, 200);

System.out.println(trainingData.head(10));

DataFrame crossValidationData = schema.build();

crossValidationData = negativeSampler.sample(crossValidationData, 40);
crossValidationData = positiveSampler.sample(crossValidationData, 40);

MultiVariateNormalOutliers method = new MultiVariateNormalOutliers();
method.fit(trainingData);

BinaryClassifierEvaluator evaluator = new BinaryClassifierEvaluator();

for(int i = 0; i < crossValidationData.rowCount(); ++i){
 boolean predicted = method.isAnomaly(crossValidationData.row(i));
 boolean actual = crossValidationData.row(i).target() > 0.5;
 evaluator.evaluate(actual, predicted);
 logger.info("predicted: {}\texpected: {}", predicted, actual);
}

evaluator.report();
```
