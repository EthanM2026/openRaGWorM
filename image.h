/*
Copyright (c) 2026 Ethan Mortonson.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef IMAGE_H
#define IMAGE_H
#include <stdlib.h>
#include <GL/glew.h>"
#include "stdio.h"

    struct _Image
    {
        GLuint Texture_ID;
        int RGBA_Image_Size;
        unsigned char Texture_Made;
        int x;
        int y;
        int Width;
        int Height;
        unsigned char* RGB_Canvas;
        unsigned char* RGBA_Canvas;
        float* Vertices;
        float* Texture_Vertices;
    };


    struct _Image* Create_Image();
    void Destroy_Image(struct _Image* Image);
    void Load_Image(struct _Image* Image, const char* Filename);
    void Render_Image(struct _Image* Image, float _x, float _y, int Alpha_Present);

    void Copy_Pixel(int Color_Depth, unsigned char* Source, unsigned char* Destination, int Source_Width, int Source_Height, int Destination_Width, int Destination_Height, int Source_X, int Source_Y, int Destination_X, int Destination_Y);
    void Copy_Pixel_Alpha(int Color_Depth, unsigned char* Source, unsigned char* Destination, int Source_Width, int Source_Height, int Destination_Width, int Destination_Height, int Source_X, int Source_Y, int Destination_X, int Destination_Y);
    void Copy_Row(int Color_Depth, unsigned char* Source, unsigned char* Destination, int Source_Width, int Source_Height, int Destination_Width, int Destination_Height, int Source_X, int Source_Y, int Destination_X, int Destination_Y, int N_Pixels_To_Copy); //do Copy Pixel Width times
    void Copy_Section(int Color_Depth, unsigned char* Source, unsigned char* Destination, int Source_Width, int Source_Height, int Destination_Width, int Destination_Height, int Source_X, int Source_Y, int Destination_X, int Destination_Y, int N_Pixels_To_Copy, int Rows_To_Copy); //do Copy_Row Height times


#endif /* IMAGE_H */



