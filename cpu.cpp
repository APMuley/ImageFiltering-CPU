#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

int main() {
    Mat img = imread("img.jpeg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "could not open image" <<std::endl;
        return -1;
    }

    auto start = high_resolution_clock::now();

    float kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1},
    };

    float kernelSum = 16;
    Mat padded;
    copyMakeBorder(img, padded, 1, 1, 1, 1, BORDER_REPLICATE);

    Mat output = Mat::zeros(img.size(), img.type());

    for (int r = 1; r < padded.rows-1; r++) {
        for (int c = 1; c < padded.cols - 1; c++) {
            // loop through kernel
            float sum = 0;
            for (int kr = -1; kr <= 1; kr++) {
                for (int kc = -1; kc <= 1; kc++) {
                    int pixel = padded.at<uchar>(kr+r, kc+c);
                    sum += pixel * kernel[kr+1][kc+1];
                }
            }

            output.at<uchar>(r-1, c-1) = (int) sum / kernelSum;
        }
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    // save image
    imwrite("gaussian.png", output);

    cout << "CPU time: " << duration.count() << " ms" << endl;

    output = Mat::zeros(img.size(), img.type());

    // Convoluted Gaussian using 2 1-D matrix

    // Allocate temp as float; same size as original image
    Mat temp = Mat::zeros(img.size(), CV_32F);

    float kernel_c[3] = {1, 2, 1};
    kernelSum = 4.0f; // sum of 1D kernel

        // Row pass
    for (int r = 1; r < padded.rows - 1; r++) {
        float* tempRow = temp.ptr<float>(r-1);
        uchar* paddedRow = padded.ptr<uchar>(r);
        for (int c = 1; c < padded.cols - 1; c++) {
            float sum = 0;
            for (int k = -1; k <= 1; k++) {
                sum += paddedRow[c + k] * kernel_c[k + 1];
            }
            tempRow[c-1] = sum / kernelSum;
        }
    }

        // Column pass
    for (int c = 1; c < padded.cols - 1; c++) {
        for (int r = 1; r < padded.rows - 1; r++) {
            float sum = 0;
            for (int k = -1; k <= 1; k++) {
                float* tempPtr = temp.ptr<float>(r + k); // row r+k-1
                sum += tempPtr[c-1] * kernel_c[k + 1];
            }
            uchar* outputRow = output.ptr<uchar>(r-1);
            outputRow[c-1] = (uchar)(sum / kernelSum + 0.5f);
        }
    }

    end = high_resolution_clock::now();

    duration = duration_cast<milliseconds>(end-start);

    cout << "2 kernel approach: " << duration.count() << "ms" << endl;

    imwrite("g3.png", output);



    return 0;
}