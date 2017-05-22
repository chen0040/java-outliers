package com.github.chen0040.outliers.guassian;


import Jama.Matrix;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import org.apache.commons.math3.distribution.NormalDistribution;


/**
 * Created by xschen on 12/8/15.
 */
public class MultiVariateNormalOutliers extends NormalOutliers {

    private Matrix matrixVar;
    private Matrix matrixMu;


    public MultiVariateNormalOutliers(){
        super();
    }



    @Override
    public double calculateProbability(DataRow tuple) {


        double[] x = tuple.toArray();

        int n = x.length;
        Matrix X = new Matrix(n, 1);
        for(int k=0; k < n; ++k){
            X.set(k, 0, x[k]);
        }

        double det = matrixVar.det();
        Matrix Var_inverse = matrixVar.inverse();
        Matrix X_minus_mu = X.minus(matrixMu);
        Matrix X_minus_mu_transpose = X_minus_mu.transpose();
        Matrix V = X_minus_mu_transpose.times(Var_inverse).times(X_minus_mu);

        double num2 = Math.pow(2 * Math.PI, n / 2.0) * Math.sqrt(Math.abs(det));

        return Math.exp(-0.5 * V.get(0, 0)) / num2;
    }

    public void fit(DataFrame batch) {

        super.fit(batch);

        int dimension = batch.row(0).toArray().length;

        matrixMu = new Matrix(dimension, 1);
        matrixVar = new Matrix(dimension, dimension);

        for(int k=0; k < dimension; ++k){
            NormalDistribution distribution = model.get(k);
            matrixMu.set(k, 0, distribution.getMean());
        }

        int batchSize = batch.rowCount();


        for(int i=0; i < batchSize; ++i){
            Matrix X = new Matrix(dimension, 1);
            DataRow tuple = batch.row(i);
            double[] x = tuple.toArray();
            for(int k=0; k < dimension; ++k){
                X.set(k, 0, x[k]);
            }
            Matrix X_minus_mu = X.minus(matrixMu);
            Matrix X_minus_mu_transpose = X_minus_mu.transpose();

            matrixVar = matrixVar.plus(X_minus_mu.times(X_minus_mu_transpose).times(1.0 / batchSize));
        }


        if(isAutoThresholding()){
            adjustThreshold(batch);
        }
    }
}
