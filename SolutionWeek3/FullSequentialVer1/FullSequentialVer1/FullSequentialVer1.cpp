#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

// ������� ��� ������������ ��������� ������� �� ����������
cv::Mat applyMedianFilter(const cv::Mat& inputImage, int kernelSize) {
    cv::Mat outputImage;
    cv::medianBlur(inputImage, outputImage, kernelSize);
    return outputImage;
}

// ������� ��� ������������ ������� ����� �� ����������
cv::Mat applyGaussianFilter(const cv::Mat& inputImage, int kernelSize, double sigma) {
    cv::Mat outputImage;
    cv::GaussianBlur(inputImage, outputImage, cv::Size(kernelSize, kernelSize), sigma);
    return outputImage;
}

// ������� ��� ������������ ������� ������ �� ����������
cv::Mat applySharpnessFilter(const cv::Mat& inputImage) {
    cv::Mat outputImage;
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0,
        -1, 5, -1,
        0, -1, 0); // ���� ��� ������� ������
    cv::filter2D(inputImage, outputImage, inputImage.depth(), kernel);
    return outputImage;
}

// ������� ��� ������������ ������� ����������� �� ������ (Unsharp Masking)
cv::Mat applyUnsharpMasking(const cv::Mat& inputImage, double sigma, double alpha) {
    cv::Mat blurred;
    cv::GaussianBlur(inputImage, blurred, cv::Size(0, 0), sigma, sigma);
    cv::Mat sharpened = inputImage + alpha * (inputImage - blurred);
    return sharpened;
}

int main() {
    for (int i = 1; i < 7; i++) {
        // ������������ ���������� � ����� JPG
        cv::Mat image = cv::imread("D:\\University\\TDS\\CourseWorkTDS\\testDataSet\\" + to_string(i) + ".jpg");

        if (image.empty()) {
            std::cerr << "������� ������������ ���������� � �����." << std::endl;
            return -1;
        }

        // ����� ����������� ����������
        cv::imshow("��������� ����������", image);
        cv::waitKey(0);

        // ������������ ��������� �������
        int medianKernelSize = 5; // ����� ���� ��������� ������� (������� �����)
        cv::Mat medianFilteredImage = applyMedianFilter(image, medianKernelSize);

        // ������������ ������� �����
        int gaussianKernelSize = 5; // ����� ���� ������� ����� (������� �����)
        double gaussianSigma = 1.0; // ���������� ��������� ��� ������� �����
        cv::Mat gaussianFilteredImage = applyGaussianFilter(medianFilteredImage, gaussianKernelSize, gaussianSigma);

        // ������������ ������� ������
        cv::Mat sharpFilteredImage = applySharpnessFilter(gaussianFilteredImage);

        // ������������ ������� ����������� �� ������ (Unsharp Masking)
        double unsharpSigma = 1.0; // ���������� ��������� ��� ������� ����������� �� ������
        double unsharpAlpha = 0.5; // �������� alpha ��� ������� ����������� �� ������
        cv::Mat unsharpMaskedImage = applyUnsharpMasking(sharpFilteredImage, unsharpSigma, unsharpAlpha);

        // ����� ����������� ����������
        cv::imshow("���������� ���� �������", unsharpMaskedImage);
        cv::waitKey(0);

        // ���������� ����������� ����������
        cv::imwrite("D:\\University\\TDS\\CourseWorkTDS\\testDataSet\\FullySequentialOutput" + to_string(i) + ".jpg", unsharpMaskedImage);
    }

    return 0;
}
