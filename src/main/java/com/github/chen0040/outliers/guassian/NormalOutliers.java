package com.github.chen0040.outliers.guassian;



import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import org.apache.commons.math3.distribution.NormalDistribution;

import java.util.*;

@Getter
@Setter
public class NormalOutliers {

    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    protected final HashMap<Integer, NormalDistribution> model = new HashMap<>();;

    private double threshold = 0.02;
    private boolean autoThresholding = false;
    private double autoThresholdingRatio = 0.05;


    public void copy(NormalOutliers that){
        model.clear();
        for(Integer i : that.model.keySet()){
            NormalDistribution d = that.model.get(i);
            model.put(i, d);
        }
    }

    public NormalOutliers makeCopy(){
        NormalOutliers clone = new NormalOutliers();
        clone.copy(this);
        return clone;
    }


    public void scratch(){
        model.clear();
    }

    public boolean isAnomaly(DataRow tuple){
        double p = calculateProbability(tuple);
        return p < threshold;
    }

    // translate the tuple into a probability value indicating the chance of the tuple being in the NORMAL class
    public double transform(DataRow tuple){
        return calculateProbability(tuple);
    }

    public double calculateProbability(DataRow tuple){
        double product = 1;
        double[] x = tuple.toArray();
        for(int i=0; i < x.length; ++i){
            product *= calculateFeatureProbability(i, x[i]);
        }
        return product;
    }

    protected double calculateFeatureProbability(int index, double value){
        NormalDistribution distribution = model.get(index);
        return distribution.probability(value);
    }

    public void fit(DataFrame batch){
        int size = batch.rowCount();
        int dimension = batch.row(0).toArray().length;

        for(int k=0; k < dimension; ++k) {
            List<Double> values = new ArrayList<>();

            for (int i = 0; i < size; ++i) {
                DataRow tuple = batch.row(i);
                double[] x = tuple.toArray();
                values.add(x[k]);
            }
            NormalDistribution distribution = model.get(k);

            if(distribution==null){
                double mu = mean(values);
                double sd = sd(values, mu);
                distribution=new NormalDistribution(mu, sd);
                model.put(k, distribution);
            }
        }

        if(isAutoThresholding()){
            adjustThreshold(batch);
        }
    }

    private double mean(List<Double> values){
        if(values.isEmpty()){
            return 0;
        }
        return values.stream().reduce((a, b) -> a + b).get() / values.size();
    }

    private double sd(List<Double> values, double mean){
        if(values.isEmpty()){
            return Double.POSITIVE_INFINITY;
        }
        double variance = values.stream().map(v -> v - mean).map(v -> v * v).reduce((a, b) -> a + b).get() / values.size();
        return Math.sqrt(variance);
    }

    protected void adjustThreshold(DataFrame batch){
        int m = batch.rowCount();

        List<Integer> orders = new ArrayList<>();
        List<Double> probs = new ArrayList<>();

        for(int i=0; i < m; ++i){
            DataRow tuple = batch.row(i);
            double prob = transform(tuple);
            probs.add(prob);
            orders.add(i);
        }

        final List<Double> probs2 = probs;
        // sort ascendingly by probability values
        Collections.sort(orders, (h1, h2) -> {
            double prob1 = probs2.get(h1);
            double prob2 = probs2.get(h2);
            return Double.compare(prob1, prob2);
        });

        int selected_index = (int)(autoThresholdingRatio * orders.size());
        if(selected_index >= orders.size()){
            setThreshold(probs.get(orders.get(orders.size()-1)));
        }
        else{
            setThreshold(probs.get(orders.get(selected_index)));
        }

    }

    public double tune(DataFrame trainingData, DataFrame crossValudation, double confidence_level){
        fit(trainingData);
        DataFrame crossValidation = trainingData.makeCopy();
        int n = crossValidation.rowCount();
        double error_rate = 1 - confidence_level;

        List<Integer> orders = new ArrayList<Integer>();
        final double[] p = new double[n];
        for(int i=0; i < n; ++i){
            p[i] = transform(crossValidation.row(i));
            orders.add(i);
        }

        // sort ascending based on the values of p
        Collections.sort(orders, (o1, o2) -> {
            double p1 = p[o1];
            double p2 = p[o2];

            return Double.compare(p1, p2);
        });

        int anomaly_count = Math.min((int) Math.ceil(error_rate * n), n);

        int anomaly_separator_index = orders.get(anomaly_count - 1);
        double best_threshold = p[anomaly_separator_index];

        this.setThreshold(best_threshold);

        return best_threshold;

    }


}
