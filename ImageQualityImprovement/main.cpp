#include <opencv2/opencv.hpp>
#include <thread>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <functional>
#include <atomic>
#include <vector>
#include <random>

auto trackTime = [](const std::function<void()>& func, const std::string& name, float& accumulator) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    //std::cout << name << " : " << duration << "\n";
    accumulator += (float) duration / 1000000;
};

float medianTime(0);
float meanTime(0);
float unsharpMaskTime(0);
float gaussianTime(0);
float combineTime(0);

cv::Mat applyMedianFilter(const cv::Mat& src, int sz) {
    cv::Mat result;
    trackTime([&]() { cv::medianBlur(src, result, sz); }, "Median Filter", medianTime);
    return result;
}

cv::Mat applyMeanFilter(const cv::Mat& src, int sz) {
    cv::Mat result;
    trackTime([&]() { cv::blur(src, result, cv::Size(sz, sz)); }, "Mean Filter", meanTime);
    return result;
}

cv::Mat applyUnsharpMask(const cv::Mat& src, int sz) {
    cv::Mat blurred, result;
    trackTime([&]() {
        cv::GaussianBlur(src, blurred, cv::Size(sz, sz), 0);
        cv::addWeighted(src, 1.5, blurred, -0.5, 0, result);
        }, "Unsharp Mask Filter", unsharpMaskTime);
    return result;
}

cv::Mat applyGaussianBlur(const cv::Mat& src, int kernelSize, double sigma) {
    cv::Mat result;
    trackTime([&]() { cv::GaussianBlur(src, result, cv::Size(kernelSize, kernelSize), sigma); }, "Gaussian Blur", gaussianTime);
    return result;
}

cv::Mat combineImages(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& img3, const cv::Mat& img4, const std::vector<double>& weights) {
    cv::Mat combinedImage;
    trackTime([&]() {
        int rows = img1.rows;
        int cols = img1.cols;
        combinedImage.create(rows, cols, CV_8UC3);

#pragma omp parallel for collapse(3)
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int c = 0; c < 3; c++) {
                    float sum = img1.at<cv::Vec3b>(i, j)[c] * weights[0] +
                        img2.at<cv::Vec3b>(i, j)[c] * weights[1] +
                        img3.at<cv::Vec3b>(i, j)[c] * weights[2] +
                        img4.at<cv::Vec3b>(i, j)[c] * weights[3];
                    combinedImage.at<cv::Vec3b>(i, j)[c] = static_cast<unsigned char>(sum);
                }
            }
        }

        }, "Combine Images", combineTime);
    return combinedImage;
}


double PSNR(const cv::Mat& image1, const cv::Mat& image2) {
    return cv::PSNR(image1, image2);
}

double SSIM(const cv::Mat& i1, const cv::Mat& i2) {
    const double C1 = 6.5025, C2 = 58.5225;
    int d = CV_32F;

    cv::Mat I1, I2;
    i1.convertTo(I1, d);           // Convert to float
    i2.convertTo(I2, d);

    cv::Mat I2_2 = I2.mul(I2);   // I2^2
    cv::Mat I1_2 = I1.mul(I1);   // I1^2
    cv::Mat I1_I2 = I1.mul(I2);   // I1 * I2

    cv::Mat mu1, mu2;
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;

    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);   // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);   // t1 = ((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map); // ssim_map = t3./t1;

    cv::Scalar mssim = cv::mean(ssim_map);
    return (mssim[0] + mssim[1] + mssim[2]) / 3; // Average over the channels
}

void addGaussianNoise(cv::Mat& image) {
    cv::Mat noise = cv::Mat(image.size(), image.type());
    std::normal_distribution<float> dist(0.0, 25.0);  // Mean and standard deviation
    std::default_random_engine generator;

    image.forEach<cv::Vec3b>([&](cv::Vec3b& pixel, const int* position) -> void {
        for (int c = 0; c < image.channels(); c++) {
            float randomValue = dist(generator);
            int adjustedValue = pixel[c] + static_cast<int>(randomValue);
            pixel[c] = cv::saturate_cast
                <unsigned char>(adjustedValue);
        }
    });
}

void addSaltAndPepperNoise(cv::Mat& image, double saltProb, double pepperProb) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    int numSalt = static_cast<int>(saltProb * image.total());
    int numPepper = static_cast<int>(pepperProb * image.total());

    for (int i = 0; i < numSalt; i++) {
        int r = distribution(generator) * image.rows;
        int c = distribution(generator) * image.cols;
        image.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);  // White
    }

    for (int i = 0; i < numPepper; i++) {
        int r = distribution(generator) * image.rows;
        int c = distribution(generator) * image.cols;
        image.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);  // Black
    }
}

double MSE(const cv::Mat& img1, const cv::Mat& img2) {
    if (img1.empty() || img2.empty() || img1.size() != img2.size() || img1.type() != img2.type()) {
        std::cerr << "Error: Images must be of the same size and type." << std::endl;
        return -1.0;
    }
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);

    cv::Scalar sumOfSquares = cv::sum(diff);
    double mse = sumOfSquares[0] + sumOfSquares[1] + sumOfSquares[2];
    mse /= (double)(img1.channels() * img1.total());

    return mse;
}

bool printImages;

void showImage(const char* windowName, cv::Mat& img) {
    if (printImages)
        cv::imshow(windowName, img);
}

bool firstTime = true;

void printResults(const cv::Mat& orig, const cv::Mat& noised, const cv::Mat& result, std::vector<std::string> filterNames) {
    
    if (firstTime) {
        std::cout << "\n";
        std::cout << "Metrics for original and noised image. MSE: " << MSE(orig, noised);
        std::cout << " PSNR: " << PSNR(orig, noised) << " SSIM: " << SSIM(orig, noised) << "\n";

        firstTime = false;
    }

    int sizeNeed = 39;
    int curr = 0;

    for (auto str : filterNames) {std::cout << str << " ";curr += str.size() + 1;}

    while (curr < sizeNeed) {std::cout << " ";curr++;}

    std::cout << "MSE: " << MSE(orig, result) << " PSNR: " << PSNR(orig, result) << " SSIM: " << SSIM(orig, result) << "\n";
    //std::cout <<MSE(orig, result) << "\n";
    //std::cout <<PSNR(orig, result) << "\n";
    //std::cout <<SSIM(orig, result) << "\n";
}

int main() {
    std::cout << "1\n2\n3\n";
    int imageI = 6;
    int width = 400;
    int height = 400;
    int threads = 1;
    omp_set_num_threads(threads); // it works 
    printImages = false;
    bool printMetrics = true;

    const char* original = "original";
    const char* gausianNoiseWindow = "gausianNoise";
    const char* result = "result";

    //std::string imagePath = "D:/T/6_term/TDS/ImageQualityImprovement/images/" + std::to_string(imageI) + ".jpg";
    //std::string imagePath = "D:/T/6_term/TDS/ImageQualityImprovement/images/good" + std::to_string(imageI) + ".jpg";
    //std::string imagePath = "D:/T/6_term/TDS/ImageQualityImprovement/images/dataset/image_" + std::to_string(imageI) + ".jpg";
    std::string imagePath = "D:/T/6_term/TDS/ImageQualityImprovement/images/dataset/big.jpg";

    cv::Mat originalImage = cv::imread(imagePath, cv::IMREAD_COLOR);
    std::cout << SSIM(originalImage, originalImage) << "\n\n"; // FOR ANNOING LOG WARNING
    if (originalImage.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }

    cv::Mat gausianNoise = originalImage.clone();
    addGaussianNoise(gausianNoise);

    if (printImages) {
        cv::namedWindow(original, cv::WINDOW_NORMAL);
        cv::namedWindow(gausianNoiseWindow, cv::WINDOW_NORMAL);
        cv::namedWindow(result, cv::WINDOW_NORMAL);
        cv::resizeWindow(original, width, height);
        cv::resizeWindow(gausianNoiseWindow, width, height);
        cv::resizeWindow(result, width, height);
    }

    showImage(original, originalImage);
    showImage(gausianNoiseWindow, gausianNoise);

    cv::Mat imageToImprove = gausianNoise.clone();

    std::cout << imageToImprove.rows << " " << imageToImprove.cols << "\n";


    float totalTimeSpentWithoutMetrics = 0;
    float totalThreadsTime = 0;
    float totalCombiningTime = 0;

    std::cout << std::fixed << std::setprecision(3);
    
    cv::Mat onlyMedian, onlyMean, onlyUnsharpMask, onlyGaussian;

    cv::Mat medianMean, medianUnsharpMask, medianGaussian;
    cv::Mat meanUnsharpMask, meanGaussian;
    cv::Mat unsharpMaskGaussian;

    cv::Mat meanUnsharpMaskGaussian;
    cv::Mat medianUnsharpMaskGaussian;
    cv::Mat medianMeanGaussian;
    cv::Mat medianMeanUnsharpMask;



    int times = 1;
    for (int iteration = 1; iteration <= times; ++iteration) {
        auto startProgram = std::chrono::high_resolution_clock::now();
        cv::Mat medianFiltered, meanFiltered, unsharpMaskFiltered, gaussianFiltered;

        std::thread threadMedian([&]() { medianFiltered = applyMedianFilter(imageToImprove, 5); });
        std::thread threadMean([&]() { meanFiltered = applyMeanFilter(imageToImprove, 5); });
        std::thread threadUnsharpMask([&]() { unsharpMaskFiltered = applyUnsharpMask(imageToImprove, 5); });
        std::thread threadGaussian([&]() { gaussianFiltered = applyGaussianBlur(imageToImprove, 5, 10); });

        threadMedian.join();
        threadMean.join();
        threadUnsharpMask.join();
        threadGaussian.join();

        auto threadsEnd = std::chrono::high_resolution_clock::now();

        std::vector<double> weights = { 0.4, 0.1, 0.4, 0.1 };
        cv::Mat combinedImage = combineImages(medianFiltered, meanFiltered, unsharpMaskFiltered, gaussianFiltered, weights);

        onlyMedian = combineImages(medianFiltered, meanFiltered, unsharpMaskFiltered, gaussianFiltered, { 1, 0, 0, 0 });
        onlyMean = combineImages(medianFiltered, meanFiltered, unsharpMaskFiltered, gaussianFiltered, { 0, 1, 0, 0 });
        onlyUnsharpMask = combineImages(medianFiltered, meanFiltered, unsharpMaskFiltered, gaussianFiltered, { 0, 0, 1, 0 });
        onlyGaussian = combineImages(medianFiltered, meanFiltered, unsharpMaskFiltered, gaussianFiltered, { 0, 0, 0, 1 });

        medianMean = combineImages(medianFiltered, meanFiltered, unsharpMaskFiltered, gaussianFiltered, { 0.5, 0.5, 0, 0 });
        medianUnsharpMask = combineImages(medianFiltered, meanFiltered, unsharpMaskFiltered, gaussianFiltered, { 0.5, 0, 0.5, 0 });
        medianGaussian = combineImages(medianFiltered, meanFiltered, unsharpMaskFiltered, gaussianFiltered, { 0.5, 0, 0, 0.5 });
        meanUnsharpMask = combineImages(medianFiltered, meanFiltered, unsharpMaskFiltered, gaussianFiltered, {0, 0.5, 0.5, 0 });
        meanGaussian = combineImages(medianFiltered, meanFiltered, unsharpMaskFiltered, gaussianFiltered, {0, 0.5, 0, 0.5 });
        unsharpMaskGaussian = combineImages(medianFiltered, meanFiltered, unsharpMaskFiltered, gaussianFiltered, {0, 0, 0.5, 0.5 });

        meanUnsharpMaskGaussian = combineImages(medianFiltered, meanFiltered, unsharpMaskFiltered, gaussianFiltered, { 0, 0.33, 0.33, 0.34 });
        medianUnsharpMaskGaussian = combineImages(medianFiltered, meanFiltered, unsharpMaskFiltered, gaussianFiltered, { 0.33, 0, 0.33,  0.34 });
        medianMeanGaussian = combineImages(medianFiltered, meanFiltered, unsharpMaskFiltered, gaussianFiltered, { 0.33, 0.33, 0,  0.34 });
        medianMeanUnsharpMask = combineImages(medianFiltered, meanFiltered, unsharpMaskFiltered, gaussianFiltered, { 0.33, 0.33,  0.34, 0 });

        printResults(originalImage, gausianNoise, onlyMedian, { "Median" });
        printResults(originalImage, gausianNoise, onlyMean, { "Mean" });
        printResults(originalImage, gausianNoise, onlyGaussian, { "Gaussian" });
        printResults(originalImage, gausianNoise, onlyUnsharpMask, { "UnsharpMask" });

        printResults(originalImage, gausianNoise, medianMean, { "Median", "Mean"});
        printResults(originalImage, gausianNoise, medianGaussian, { "Median", "Gaussian" });
        printResults(originalImage, gausianNoise, medianUnsharpMask, { "Median", "UnsharpMask"});
        printResults(originalImage, gausianNoise, meanGaussian, { "Mean", "Gaussian" });
        printResults(originalImage, gausianNoise, meanUnsharpMask, { "Mean", "UnsharpMask"});
        printResults(originalImage, gausianNoise, unsharpMaskGaussian, { "Gaussian", "UnsharpMask"});


        printResults(originalImage, gausianNoise, meanUnsharpMaskGaussian, { "Mean", "UnsharpMask", "Gaussian"});
        printResults(originalImage, gausianNoise, medianUnsharpMaskGaussian, { "Median", "UnsharpMask", "Gaussian"});
        printResults(originalImage, gausianNoise, medianMeanGaussian, { "Median", "Mean", "Gaussian"});
        printResults(originalImage, gausianNoise, medianMeanUnsharpMask, { "Median", "Mean", "UnsharpMask" });

        printResults(originalImage, gausianNoise, combinedImage, { "Median", "Mean", "Gaussian", "UnsharpMask" });
        
        showImage(result, combinedImage);

        auto endProgram = std::chrono::high_resolution_clock::now();
        auto totalIterationTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endProgram - startProgram).count();
        totalTimeSpentWithoutMetrics += (float) totalIterationTime / 1000000;
        
        auto iterationThreadsTime = std::chrono::duration_cast<std::chrono::nanoseconds>(threadsEnd - startProgram).count();
        totalThreadsTime += (float)iterationThreadsTime / 1000000;

        auto iterationCombiningTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endProgram - threadsEnd).count();
        totalCombiningTime += (float)iterationCombiningTime / 1000000;

        if (printMetrics) {
            std::cout << "\nMSE original and noised: " << MSE(originalImage, gausianNoise);
            std::cout << "\nPSNR original and noised: " << PSNR(originalImage, gausianNoise);
            std::cout << "\nSSIM original and noised: " << SSIM(originalImage, gausianNoise);

            std::cout << "\nMSEoriginal and improved: " << MSE(originalImage, combinedImage);
            std::cout << "\nPSNR original and improved: " << PSNR(originalImage, combinedImage);
            std::cout << "\nSSIM original and improved: " << SSIM(originalImage, combinedImage);
        }

    }

    std::cout << "\n\nthreads: " << threads << "\n";
    
    std::cout << "\n\nTotal threads runtime: " << totalThreadsTime/times << " milliseconds" << std::endl;
    std::cout << "Total combining runtime: " << totalCombiningTime/times << " milliseconds" << std::endl;
    std::cout << "Total program runtime: " << totalTimeSpentWithoutMetrics/times << " milliseconds" << std::endl;

    float totalSeqThreads = medianTime + meanTime + unsharpMaskTime + gaussianTime;
    std::cout << "\nTotal threads runtime if sequential: " << totalSeqThreads/times << " milliseconds" << std::endl;
    std::cout << "Total combining runtime if sequential: " << combineTime/times << " milliseconds" << std::endl;
    std::cout << "Total program runtime if sequential: " << (totalSeqThreads + combineTime)/times << " milliseconds" << std::endl;

    std::cout << "\nTotal time for Median Filter: " << medianTime/times << " milliseconds" << std::endl;
    std::cout << "Total time for Mean Filter: " << meanTime/times << " milliseconds" << std::endl;
    std::cout << "Total time for Unsharp Mask Filter: " << unsharpMaskTime/times << " milliseconds" << std::endl;
    std::cout << "Total time for Gaussian Blur: " << gaussianTime/times << " milliseconds" << std::endl;
    std::cout << "Total time for Combining Images: " << combineTime/times << " milliseconds" << std::endl;


    float totalSeq = medianTime + meanTime + unsharpMaskTime + gaussianTime + combineTime;
    std::cout << "\n\nTotal time if sequential: " << totalSeq/times << " milliseconds" << std::endl;
    /// FOR EXACT RESULTS RUN PROGRAM WITH THREADS = 1

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
