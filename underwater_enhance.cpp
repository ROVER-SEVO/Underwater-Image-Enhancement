#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;


// Global Variables (for Trackbar callbacks)
Mat g_src, g_dst;
int g_a_shift = 50;      // 0-100 mapped to -50 to 50
int g_b_shift = 50;      // 0-100 mapped to -50 to 50
int g_omega = 75;        // 0-100 mapped to 0.0 - 1.0
int g_clahe_clip = 12;   // 0-50 mapped to 0.0 - 5.0
int g_red_strength = 30; // 0-100
const float T_MIN = 0.35f; // Minimum transmission


// Helper: Calculate Percentile
double getPercentile(const Mat& gray, double percentile) {
    // Flatten and sort to find percentile
    Mat flat;
    gray.reshape(1, 1).copyTo(flat);
    cv::sort(flat, flat, SORT_EVERY_ROW + SORT_ASCENDING);
    
    int index = (int)(flat.cols * percentile);
    if (index >= flat.cols) index = flat.cols - 1;
    
    return (double)flat.at<uchar>(0, index);
}

// Core Processing Functions

// 2.1 White Balance (LAB Color Space)
Mat white_balance(const Mat& img, int a_shift, int b_shift) {
    Mat lab;
    cvtColor(img, lab, COLOR_BGR2Lab);
    
    vector<Mat> channels;
    split(lab, channels); // L, A, B
    
    // Convert to float for calculation
    Mat A, B;
    channels[1].convertTo(A, CV_32F);
    channels[2].convertTo(B, CV_32F);
    
    // Calculate mean
    Scalar meanA = mean(A);
    Scalar meanB = mean(B);
    
    // Apply shift offset
    A = A - (meanA[0] - 128.0) + a_shift;
    B = B - (meanB[0] - 128.0) + b_shift;
    
    // Convert back to 8U and clip
    A.convertTo(channels[1], CV_8U);
    B.convertTo(channels[2], CV_8U);
    
    Mat result;
    merge(channels, lab);
    cvtColor(lab, result, COLOR_Lab2BGR);
    return result;
}

// 2.2 Red Channel Restoration
Mat restore_red(const Mat& img, int strength_val) {
    vector<Mat> channels;
    split(img, channels); // B, G, R
    
    Mat r = channels[2];
    Mat r_boost;
    equalizeHist(r, r_boost);
    
    float strength = strength_val / 100.0f;
    
    Mat r_new;
    // Blend: r_new = r * (1-strength) + r_boost * strength
    addWeighted(r, 1.0 - strength, r_boost, strength, 0, r_new);
    
    channels[2] = r_new;
    Mat result;
    merge(channels, result);
    return result;
}

// 2.3 CLAHE
Mat clahe_enhance(const Mat& img, int clip_val) {
    Mat lab;
    cvtColor(img, lab, COLOR_BGR2Lab);
    
    vector<Mat> channels;
    split(lab, channels);
    
    float limit = max(clip_val / 10.0f, 0.1f);
    Ptr<CLAHE> clahe = createCLAHE(limit, Size(8, 8));
    
    Mat L2;
    clahe->apply(channels[0], L2);
    channels[0] = L2;
    
    Mat result;
    merge(channels, lab);
    cvtColor(lab, result, COLOR_Lab2BGR);
    return result;
}

// 2.4 Dehaze (Dark Channel Prior)
Mat dehaze(const Mat& img, int omega_val) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    
    // 1. Estimate atmospheric light (top 5% percentile)
    double A_light = getPercentile(gray, 0.95);
    if (A_light < 1.0) A_light = 1.0;
    
    float omega = omega_val / 100.0f;
    
    // 2. Calculate transmission t = 1 - omega * (gray / A)
    Mat grayF;
    gray.convertTo(grayF, CV_32F);
    
    Mat t = 1.0f - omega * (grayF / (float)A_light);
    
    // Clamp transmission to lower bound (T_MIN)
    Mat t_clamped;
    cv::max(t, T_MIN, t_clamped); 
    
    // 3. Recover scene radiance J = (I - A)/t + A
    Mat imgF;
    img.convertTo(imgF, CV_32F);
    
    vector<Mat> channels;
    split(imgF, channels);
    
    for (int i = 0; i < 3; i++) {
        // (Channel - A) / t + A
        channels[i] = (channels[i] - A_light) / t_clamped + A_light;
    }
    
    Mat resultF;
    merge(channels, resultF);
    
    Mat result;
    resultF.convertTo(result, CV_8U); // Auto-clip to 0-255
    return result;
}

// 2.5 Sharpening
Mat sharpen(const Mat& img) {
    Mat blur;
    GaussianBlur(img, blur, Size(3, 3), 0);
    Mat result;
    // Unsharp Mask: 1.2*Original - 0.2*Blur
    addWeighted(img, 1.2, blur, -0.2, 0, result);
    return result;
}

// 2.6 Gamma Correction
Mat gamma_correct(const Mat& img, float g = 1.1f) {
    Mat lut(1, 256, CV_8U);
    uchar* p = lut.ptr();
    float inv = 1.0f / g;
    
    for (int i = 0; i < 256; i++) {
        p[i] = saturate_cast<uchar>(pow(i / 255.0f, inv) * 255.0f);
    }
    
    Mat result;
    LUT(img, lut, result);
    return result;
}

// Pipeline Wrapper
void process_pipeline() {
    if (g_src.empty()) return;
    
    // Map trackbar values to parameters
    int a_shift = g_a_shift - 50;
    int b_shift = g_b_shift - 50;
    
    Mat temp;
    temp = white_balance(g_src, a_shift, b_shift);
    temp = restore_red(temp, g_red_strength);
    temp = clahe_enhance(temp, g_clahe_clip);
    temp = dehaze(temp, g_omega);
    temp = sharpen(temp);
    g_dst = gamma_correct(temp, 1.1f);
    
    imshow("Enhanced", g_dst);
}

// Trackbar Callback
void on_trackbar(int, void*) {
    process_pipeline();
}

int main(int argc, char** argv) {
    // 1. Read Image
    string path = "underwater.jpg";
    
    g_src = imread(path);
    if (g_src.empty()) {
        cout << "Error: Could not open image: " << path << endl;
        cout << "Usage: ./EnhanceApp <image_path>" << endl;
        return -1;
    }

    // Scale down if image is too large for display
    if (g_src.cols > 1000) {
        double scale = 1000.0 / g_src.cols;
        resize(g_src, g_src, Size(), scale, scale);
    }

    // 2. Create Windows
    namedWindow("Original", WINDOW_AUTOSIZE);
    namedWindow("Enhanced", WINDOW_AUTOSIZE);

    imshow("Original", g_src);

    // 3. Create Trackbars
    createTrackbar("A Shift", "Enhanced", &g_a_shift, 100, on_trackbar);
    createTrackbar("B Shift", "Enhanced", &g_b_shift, 100, on_trackbar);
    createTrackbar("Omega", "Enhanced", &g_omega, 100, on_trackbar);
    createTrackbar("CLAHE", "Enhanced", &g_clahe_clip, 50, on_trackbar);
    createTrackbar("Red Boost", "Enhanced", &g_red_strength, 100, on_trackbar);

    // 4. Initial Run
    process_pipeline();

    cout << "Controls:" << endl;
    cout << "  Press 's' to save 'result.jpg'" << endl;
    cout << "  Press 'q' or ESC to quit" << endl;

    // 5. Event Loop
    while (true) {
        char key = (char)waitKey(30);
        if (key == 'q' || key == 27) {
            break;
        }
        if (key == 's') {
            imwrite("result.jpg", g_dst);
            cout << "Saved result.jpg" << endl;
        }
    }

    return 0;
}