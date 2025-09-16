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
  
    // addition of sobel edge detection
    int sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1},
    };

    int sobel_y[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1},
    };


    float kernelSum = 16;
    Mat padded;
    copyMakeBorder(img, padded, 1, 1, 1, 1, BORDER_REPLICATE);

    Mat output = Mat::zeros(img.size(), img.type());

    for (int r = 1; r < padded.rows-1; r++) {
        for (int c = 1; c < padded.cols - 1; c++) {
            // loop through kernel
            float Gx = 0, Gy = 0;
            for (int kr = -1; kr <= 1; kr++) {
                for (int kc = -1; kc <= 1; kc++) {
                    int pixel = padded.at<uchar>(kr+r, kc+c);
                    Gx += pixel * sobel_x[kr+1][kc+1];
                    Gy += pixel * sobel_y[kr+1][kc+1];
                }
            }
            // calcualte magnitude of gradient
            float mag = sqrt(Gx*Gx + Gy*Gy);

            // setting threshold to 100
            float threshold = 50.0f;
            if (mag > threshold) mag = 255.0f;
            else mag = 0.0f;

            uchar pixel_output = (uchar)min(255.0f, mag);

            output.at<uchar>(r-1, c-1) = pixel_output;
        }
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    // save image
    imwrite("gaussian.png", output);

    cout << "CPU time: " << duration.count() << " ms" << endl;


    return 0;
}