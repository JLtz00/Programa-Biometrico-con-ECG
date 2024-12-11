#ifndef SVM_CLASSIFIER_H
#define SVM_CLASSIFIER_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace std;

class SVMClassifier {
public:
    SVMClassifier() : svm(cv::ml::SVM::create()) {
        svm->setType(cv::ml::SVM::C_SVC);
        svm->setKernel(cv::ml::SVM::RBF);
        svm->setC(1.0);
        svm->setGamma(0.5);
    }

    void train(const vector<vector<double>>& trainingData, const vector<int>& labels) {
        cv::Mat trainingDataMat(trainingData.size(), trainingData[0].size(), CV_32F);
        for (int i = 0; i < trainingData.size(); i++) {
            for (int j = 0; j < trainingData[i].size(); j++) {
                trainingDataMat.at<float>(i, j) = trainingData[i][j];
            }
        }
        
        cv::Mat labelsMat(labels.size(), 1, CV_32S);
        for (int i = 0; i < labels.size(); i++) {
            labelsMat.at<int>(i, 0) = labels[i];
        }

        svm->train(trainingDataMat, cv::ml::ROW_SAMPLE, labelsMat);
    }

    int predict(const vector<double>& features) {
        cv::Mat featureMat(1, features.size(), CV_32F);
        for (int i = 0; i < features.size(); i++) {
            featureMat.at<float>(0, i) = features[i];
        }
        return static_cast<int>(svm->predict(featureMat));
    }

private:
    cv::Ptr<cv::ml::SVM> svm;
};

#endif
