// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with common samples functionality
 * @file common.hpp
 */

#pragma once

#include <string>
#include <algorithm>
#include <map>
#include <vector>
#include <list>
#include <functional>
#include <chrono>
#include <fstream>

/**
 * @brief Trims from both ends (in place)
 * @param s - string to trim
 * @return trimmed string
 */
/*inline std::string& trim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}
*/
/**
 * @brief Gets filename without extension
 * @param filepath - full file name
 * @return filename without extension
 */
static std::string fileNameNoExt(const std::string &filepath) {
    auto pos = filepath.rfind('.');
    if (pos == std::string::npos) return filepath;
    return filepath.substr(0, pos);
}

/**
* @brief Get extension from filename
* @param filename - name of the file which extension should be extracted
* @return string with extracted file extension
*/
inline std::string fileExt(const std::string &filename) {
    auto pos = filename.rfind('.');
    if (pos == std::string::npos) return "";
    return filename.substr(pos + 1);
}





inline double getDurationOf(std::function<void()> func) {
    auto t0 = std::chrono::high_resolution_clock::now();
    func();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> fs = t1 - t0;
    return std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(fs).count();
}






/**
 * @class Color
 * @brief A Color class stores channels of a given color
 */
class Color {
private:
    unsigned char _r;
    unsigned char _g;
    unsigned char _b;

public:
    /**
     * A default constructor.
     * @param r - value for red channel
     * @param g - value for green channel
     * @param b - value for blue channel
     */
    Color(unsigned char r,
          unsigned char g,
          unsigned char b) : _r(r), _g(g), _b(b) {}

    inline unsigned char red() {
        return _r;
    }

    inline unsigned char blue() {
        return _b;
    }

    inline unsigned char green() {
        return _g;
    }
};

/**
* @brief Adds colored rectangles to the image
* @param data - data where rectangles are put
* @param height - height of the rectangle
* @param width - width of the rectangle
* @param rectangles - vector points for the rectangle, should be 4x compared to num classes
* @param classes - vector of classes
* @param thickness - thickness of a line (in pixels) to be used for bounding boxes
*/
static void addRectangles(unsigned char *data, size_t height, size_t width, std::vector<int> rectangles, std::vector<int> classes, int thickness = 1) {
    std::vector<Color> colors = {  // colors to be used for bounding boxes
        { 128, 64,  128 },
        { 232, 35,  244 },
        { 70,  70,  70 },
        { 156, 102, 102 },
        { 153, 153, 190 },
        { 153, 153, 153 },
        { 30,  170, 250 },
        { 0,   220, 220 },
        { 35,  142, 107 },
        { 152, 251, 152 },
        { 180, 130, 70 },
        { 60,  20,  220 },
        { 0,   0,   255 },
        { 142, 0,   0 },
        { 70,  0,   0 },
        { 100, 60,  0 },
        { 90,  0,   0 },
        { 230, 0,   0 },
        { 32,  11,  119 },
        { 0,   74,  111 },
        { 81,  0,   81 }
    };
    if (rectangles.size() % 4 != 0 || rectangles.size() / 4 != classes.size()) {
        return;
    }

    for (size_t i = 0; i < classes.size(); i++) {
        int x = rectangles.at(i * 4);
        int y = rectangles.at(i * 4 + 1);
        int w = rectangles.at(i * 4 + 2);
        int h = rectangles.at(i * 4 + 3);

        int cls = classes.at(i) % colors.size();  // color of a bounding box line

        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (w < 0) w = 0;
        if (h < 0) h = 0;

        if (static_cast<std::size_t>(x) >= width) { x = width - 1; w = 0; thickness = 1; }
        if (static_cast<std::size_t>(y) >= height) { y = height - 1; h = 0; thickness = 1; }

        if (static_cast<std::size_t>(x + w) >= width) { w = width - x - 1; }
        if (static_cast<std::size_t>(y + h) >= height) { h = height - y - 1; }

        thickness = std::min(std::min(thickness, w / 2 + 1), h / 2 + 1);

        size_t shift_first;
        size_t shift_second;
        for (int t = 0; t < thickness; t++) {
            shift_first = (y + t) * width * 3;
            shift_second = (y + h - t) * width * 3;
            for (int ii = x; ii < x + w + 1; ii++) {
                data[shift_first + ii * 3] = colors.at(cls).red();
                data[shift_first + ii * 3 + 1] = colors.at(cls).green();
                data[shift_first + ii * 3 + 2] = colors.at(cls).blue();
                data[shift_second + ii * 3] = colors.at(cls).red();
                data[shift_second + ii * 3 + 1] = colors.at(cls).green();
                data[shift_second + ii * 3 + 2] = colors.at(cls).blue();
            }
        }

        for (int t = 0; t < thickness; t++) {
            shift_first = (x + t) * 3;
            shift_second = (x + w - t) * 3;
            for (int ii = y; ii < y + h + 1; ii++) {
                data[shift_first + ii * width * 3] = colors.at(cls).red();
                data[shift_first + ii * width * 3 + 1] = colors.at(cls).green();
                data[shift_first + ii * width * 3 + 2] = colors.at(cls).blue();
                data[shift_second + ii * width * 3] = colors.at(cls).red();
                data[shift_second + ii * width * 3 + 1] = colors.at(cls).green();
                data[shift_second + ii * width * 3 + 2] = colors.at(cls).blue();
            }
        }
    }
}

/**
* @brief Writes output data to BMP image
* @param name - image name
* @param data - output data
* @param height - height of the target image
* @param width - width of the target image
* @return false if error else true
*/
static bool writeOutputBmp(std::string name, unsigned char *data, size_t height, size_t width) {
    std::ofstream outFile;
    outFile.open(name, std::ofstream::binary);
    if (!outFile.is_open()) {
        return false;
    }

    unsigned char file[14] = {
        'B', 'M',           // magic
        0, 0, 0, 0,         // size in bytes
        0, 0,               // app data
        0, 0,               // app data
        40 + 14, 0, 0, 0      // start of data offset
    };
    unsigned char info[40] = {
        40, 0, 0, 0,        // info hd size
        0, 0, 0, 0,         // width
        0, 0, 0, 0,         // height
        1, 0,               // number color planes
        24, 0,              // bits per pixel
        0, 0, 0, 0,         // compression is none
        0, 0, 0, 0,         // image bits size
        0x13, 0x0B, 0, 0,   // horz resolution in pixel / m
        0x13, 0x0B, 0, 0,   // vert resolution (0x03C3 = 96 dpi, 0x0B13 = 72 dpi)
        0, 0, 0, 0,         // #colors in palette
        0, 0, 0, 0,         // #important colors
    };

       if (height > (size_t)std::numeric_limits<int32_t>::max || width > (size_t)std::numeric_limits<int32_t>::max) {
           // "File size is too big: " << height << " X " << width
           return false;
    }

    int padSize = static_cast<int>(4 - (width * 3) % 4) % 4;
    int sizeData = static_cast<int>(width * height * 3 + height * padSize);
    int sizeAll = sizeData + sizeof(file) + sizeof(info);

    file[2] = (unsigned char)(sizeAll);
    file[3] = (unsigned char)(sizeAll >> 8);
    file[4] = (unsigned char)(sizeAll >> 16);
    file[5] = (unsigned char)(sizeAll >> 24);

    info[4] = (unsigned char)(width);
    info[5] = (unsigned char)(width >> 8);
    info[6] = (unsigned char)(width >> 16);
    info[7] = (unsigned char)(width >> 24);

    int32_t negativeHeight = -(int32_t)height;
    info[8] = (unsigned char)(negativeHeight);
    info[9] = (unsigned char)(negativeHeight >> 8);
    info[10] = (unsigned char)(negativeHeight >> 16);
    info[11] = (unsigned char)(negativeHeight >> 24);

    info[20] = (unsigned char)(sizeData);
    info[21] = (unsigned char)(sizeData >> 8);
    info[22] = (unsigned char)(sizeData >> 16);
    info[23] = (unsigned char)(sizeData >> 24);

    outFile.write(reinterpret_cast<char *>(file), sizeof(file));
    outFile.write(reinterpret_cast<char *>(info), sizeof(info));

    unsigned char pad[3] = { 0, 0, 0 };

        for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            unsigned char pixel[3];
            pixel[0] = data[y * width * 3 + x * 3];
            pixel[1] = data[y * width * 3 + x * 3 + 1];
            pixel[2] = data[y * width * 3 + x * 3 + 2];

            outFile.write(reinterpret_cast<char *>(pixel), 3);
        }
        outFile.write(reinterpret_cast<char *>(pad), padSize);
    }
    return true;
}
