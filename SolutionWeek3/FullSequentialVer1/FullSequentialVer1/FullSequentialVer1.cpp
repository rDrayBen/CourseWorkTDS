#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

// Функція для застосування медіанного фільтра до зображення
cv::Mat applyMedianFilter(const cv::Mat& inputImage, int kernelSize) {
    cv::Mat outputImage;
    cv::medianBlur(inputImage, outputImage, kernelSize);
    return outputImage;
}

// Функція для застосування фільтра Гауса до зображення
cv::Mat applyGaussianFilter(const cv::Mat& inputImage, int kernelSize, double sigma) {
    cv::Mat outputImage;
    cv::GaussianBlur(inputImage, outputImage, cv::Size(kernelSize, kernelSize), sigma);
    return outputImage;
}

// Функція для застосування фільтра різкості до зображення
cv::Mat applySharpnessFilter(const cv::Mat& inputImage) {
    cv::Mat outputImage;
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0,
        -1, 5, -1,
        0, -1, 0); // Ядро для фільтра різкості
    cv::filter2D(inputImage, outputImage, inputImage.depth(), kernel);
    return outputImage;
}

// Функція для застосування фільтра розкладання за гіршими (Unsharp Masking)
cv::Mat applyUnsharpMasking(const cv::Mat& inputImage, double sigma, double alpha) {
    cv::Mat blurred;
    cv::GaussianBlur(inputImage, blurred, cv::Size(0, 0), sigma, sigma);
    cv::Mat sharpened = inputImage + alpha * (inputImage - blurred);
    return sharpened;
}

int main() {
    for (int i = 1; i < 7; i++) {
        // Завантаження зображення з файлу JPG
        cv::Mat image = cv::imread("D:\\University\\TDS\\CourseWorkTDS\\testDataSet\\" + to_string(i) + ".jpg");

        if (image.empty()) {
            std::cerr << "Помилка завантаження зображення з файлу." << std::endl;
            return -1;
        }

        // Показ початкового зображення
        cv::imshow("Початкове зображення", image);
        cv::waitKey(0);

        // Застосування медіанного фільтра
        int medianKernelSize = 5; // Розмір ядра медіанного фільтра (непарне число)
        cv::Mat medianFilteredImage = applyMedianFilter(image, medianKernelSize);

        // Застосування фільтра Гауса
        int gaussianKernelSize = 5; // Розмір ядра фільтра Гауса (непарне число)
        double gaussianSigma = 1.0; // Стандартне відхилення для фільтра Гауса
        cv::Mat gaussianFilteredImage = applyGaussianFilter(medianFilteredImage, gaussianKernelSize, gaussianSigma);

        // Застосування фільтра різкості
        cv::Mat sharpFilteredImage = applySharpnessFilter(gaussianFilteredImage);

        // Застосування фільтра розкладання за гіршими (Unsharp Masking)
        double unsharpSigma = 1.0; // Стандартне відхилення для фільтра розкладання за гіршими
        double unsharpAlpha = 0.5; // Параметр alpha для фільтра розкладання за гіршими
        cv::Mat unsharpMaskedImage = applyUnsharpMasking(sharpFilteredImage, unsharpSigma, unsharpAlpha);

        // Показ обробленого зображення
        cv::imshow("Зображення після фільтрів", unsharpMaskedImage);
        cv::waitKey(0);

        // Збереження обробленого зображення
        cv::imwrite("D:\\University\\TDS\\CourseWorkTDS\\testDataSet\\FullySequentialOutput" + to_string(i) + ".jpg", unsharpMaskedImage);
    }

    return 0;
}
