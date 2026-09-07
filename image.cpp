/*
Copyright (c) 2026 Ethan Mortonson.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "image.h"
#include "string.h"

struct _Image* Create_Image()
{
    struct _Image* I = (struct _Image*) calloc(1,sizeof (struct _Image));

    I->x = 0;
    I->y = 0;
    I->Texture_Made = 0;
    return I;
}

void Destroy_Image(struct _Image* Image)
{
    //free(Image->RGB_Canvas);
    //free(Image->RGBA_Canvas);
    //free(Image);
}


void Load_Image(struct _Image* Image, const char* Filename)
{
    FILE * file;
    file = fopen(Filename,"rb");
    if(!file)
    {
        file = fopen("resources/artwork/placeholders/no_image.bmp", "rb");
    }

    unsigned char header[54];

    fread(header, 1, 54, file);

    Image->Width = *(int*)&(header[0x12]);
    Image->Height = *(int*)&(header[0x16]);

    int RGB_Image_Size = ((Image->Width) * (Image->Height) * 3);

    int RGBA_Image_Size = ((Image->Width) * (Image->Height) * 4);

    fseek(file, 54, SEEK_SET);

    Image->RGB_Canvas = (unsigned char*) malloc (RGB_Image_Size * sizeof(unsigned char));



    Image->RGBA_Canvas = (unsigned char*) malloc (RGBA_Image_Size * sizeof(unsigned char));
    memset(Image->RGBA_Canvas, 0, RGBA_Image_Size);
    fread(Image->RGB_Canvas,1,RGB_Image_Size,file);
    fclose(file);
    //Image->RGBA_Canvas = Convert_RGB_To_RGBA(Image->RGB_Canvas,Image->Width, Image->Height);
    int in = 0;
    int r = 0;
    //Image->RGBA_Canvas[0] = Image->RGB_Canvas[0];
    for(int j = 0; j < (RGBA_Image_Size/4); j++)
    {

        Image->RGBA_Canvas[r] = Image->RGB_Canvas[in+2];
        Image->RGBA_Canvas[r + 1] = Image->RGB_Canvas[in + 1];
        Image->RGBA_Canvas[r + 2] = Image->RGB_Canvas[in];
        Image->RGBA_Canvas[r + 3] = 255;
        if(Image->RGBA_Canvas[r] == 200)
        {
            if(Image->RGBA_Canvas[r+1] == 200)
            {
                if(Image->RGBA_Canvas[r + 2] == 200)
                {
                    Image->RGBA_Canvas[r + 3] = 0;
                }
            }
        }
        in += 3;
        r += 4;


    }

    // Generate a texture ID
    glGenTextures(1, &Image->Texture_ID);
    // Bind the texture to GL_TEXTURE_2D
    glBindTexture(GL_TEXTURE_2D, Image->Texture_ID);

    // Set texture parameters
    // GL_LINEAR for smooth scaling, GL_NEAREST for pixelated look
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Clamp to edge to avoid issues with non-power-of-2 textures
   // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  //  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Load the image data into the texture
    // GL_BGR is used because BMP stores pixels in BGR order
    // GL_UNSIGNED_BYTE indicates each color component is an unsigned byte
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, Image->Width, Image->Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, Image->RGBA_Canvas);

    // Enable texture mapping
    glEnable(GL_TEXTURE_2D);

}

void Render_Image(struct _Image* Image, float _x, float _y, int Alpha_Present)
{
    glBindTexture(GL_TEXTURE_2D, Image->Texture_ID);

    float Vertices[] = {_x, _y, _x + Image->Width, _y, _x + Image->Width, _y + Image->Height, _x, _y + Image->Height};
    float Texture_Vertices[] = {0, 0, -1, 0, -1, 1, 0, 1};
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY_EXT);
    glVertexPointer(2, GL_FLOAT, 0, Vertices);
    glTexCoordPointer(2, GL_FLOAT, 0, Texture_Vertices);
    glDrawArrays(GL_QUADS, 0, 4);
    glDisableClientState(GL_VERTEX_ARRAY);
}


void Copy_Pixel(int Color_Depth, unsigned char* Source, unsigned char* Destination, int Source_Width, int Source_Height, int Destination_Width, int Destination_Height, int Source_X, int Source_Y, int Destination_X, int Destination_Y)
{
/* TIMES 3!??
 * COPY PIXELS
 * 1. Calculate Source and Destination array Sizes
 * 2. Calculate Source and Destination array locations
 * 3. Send Pixel
 */
    int Source_Array_Size;
    int Destination_Array_Size;
    int Source_Array_Location;
    int Destination_Array_Location;

    Source_Array_Size = Source_Width * Source_Height * Color_Depth;
    Destination_Array_Size = Destination_Width * Destination_Height * Color_Depth;

    Source_Array_Location = (Source_Y * ( Source_Width * Color_Depth)) + (Source_X * Color_Depth);
    Destination_Array_Location = (Destination_Y * ( Destination_Width * Color_Depth)) + (Destination_X * Color_Depth);

    if(Color_Depth == 1)
    {
        memcpy(&Destination[Destination_Array_Location], &Source[Source_Array_Location], 1);

        //Destination[Destination_Array_Location] = Source[Source_Array_Location];
    }

    else if(Color_Depth != 1)
    {
        memcpy(&Destination[Destination_Array_Location], &Source[Source_Array_Location], 1);
        memcpy(&Destination[Destination_Array_Location+1], &Source[Source_Array_Location+1], 1);
        memcpy(&Destination[Destination_Array_Location+2], &Source[Source_Array_Location+2], 1);
    }
}

void Copy_Pixel_Alpha(int Color_Depth, unsigned char* Source, unsigned char* Destination, int Source_Width, int Source_Height, int Destination_Width, int Destination_Height, int Source_X, int Source_Y, int Destination_X, int Destination_Y)
{
    int Source_Array_Size;
    int Destination_Array_Size;
    int Source_Array_Location;
    int Destination_Array_Location;

    Source_Array_Size = Source_Width * Source_Height * Color_Depth;
    Destination_Array_Size = Destination_Width * Destination_Height * Color_Depth;

    Source_Array_Location = (Source_Y * ( Source_Width * Color_Depth)) + (Source_X * Color_Depth);
    Destination_Array_Location = (Destination_Y * ( Destination_Width * Color_Depth)) + (Destination_X * Color_Depth);

    //////printf("Source Array Location: %d\n",Source_Array_Location);
    //////printf("Dest Array Location: %d\n",Destination_Array_Location);

    if(Color_Depth == 1)
    {
        memcpy(&Destination[Destination_Array_Location], &Source[Source_Array_Location], 1);
        //intf("The color depth is 1\n");

        //Destination[Destination_Array_Location] = Source[Source_Array_Location];
    }

    else if(Color_Depth != 1)
    {
        memcpy(&Destination[Destination_Array_Location], &Source[Source_Array_Location], 1);
        memcpy(&Destination[Destination_Array_Location+1], &Source[Source_Array_Location+1], 1);
        memcpy(&Destination[Destination_Array_Location+2], &Source[Source_Array_Location+2], 1);
        memcpy(&Destination[Destination_Array_Location+3], &Source[Source_Array_Location+3], 1);
       // //printf("The color depth is 4\n");
    }
}

void Copy_Row(int Color_Depth, unsigned char* Source, unsigned char* Destination, int Source_Width, int Source_Height, int Destination_Width, int Destination_Height, int Source_X, int Source_Y, int Destination_X, int Destination_Y, int N_Pixels_To_Copy)
{
    int Line_Counter = 0;
    while(Line_Counter < N_Pixels_To_Copy)
    {
        if(Color_Depth == 4)
        {
        Copy_Pixel_Alpha(4,Source,Destination,Source_Width,Source_Height,Destination_Width,Destination_Height,Source_X+Line_Counter,Source_Y,Destination_X+Line_Counter,Destination_Y);
        Line_Counter += 1;
        }

        else if(Color_Depth == 3)
        {
        Copy_Pixel(3,Source,Destination,Source_Width,Source_Height,Destination_Width,Destination_Height,Source_X+Line_Counter,Source_Y,Destination_X+Line_Counter,Destination_Y);
        Line_Counter += 1;
        }

        else if(Color_Depth == 1)
        {
        Copy_Pixel(1,Source,Destination,Source_Width,Source_Height,Destination_Width,Destination_Height,Source_X+Line_Counter,Source_Y,Destination_X+Line_Counter,Destination_Y);
        Line_Counter += 1;
        }

    }
}

void Copy_Section(int Color_Depth, unsigned char* Source, unsigned char* Destination, int Source_Width, int Source_Height, int Destination_Width, int Destination_Height, int Source_X, int Source_Y, int Destination_X, int Destination_Y, int N_Pixels_To_Copy, int Rows_To_Copy)
{
    ////printf("The color depth is %d\n", Color_Depth);
    int Row_Counter = 0;
    while(Row_Counter < Rows_To_Copy)
    {
        if(Color_Depth == 4)
        {
        Copy_Row(4,Source,Destination,Source_Width,Source_Height,Destination_Width,Destination_Height,Source_X,Source_Y+Row_Counter,Destination_X,Destination_Y+Row_Counter,N_Pixels_To_Copy);
        Row_Counter += 1;
        }

        else if(Color_Depth == 3)
        {
        Copy_Row(3,Source,Destination,Source_Width,Source_Height,Destination_Width,Destination_Height,Source_X,Source_Y+Row_Counter,Destination_X,Destination_Y+Row_Counter,N_Pixels_To_Copy);
        Row_Counter += 1;
        }

        else if(Color_Depth == 1)
        {
        Copy_Row(1,Source,Destination,Source_Width,Source_Height,Destination_Width,Destination_Height,Source_X,Source_Y+Row_Counter,Destination_X,Destination_Y+Row_Counter,N_Pixels_To_Copy);
        Row_Counter += 1;
        }
    }
}
