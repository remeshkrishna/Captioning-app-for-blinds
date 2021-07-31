package com.example.project;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import com.google.gson.Gson;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    private ImageView imgview;
    private Button select,predict;
    private TextView tv;
    private Bitmap img;
    Interpreter interpreter;
    Interpreter encoder;
    Interpreter decoder;
    TextToSpeech t1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imgview = (ImageView)findViewById(R.id.imageView);
        select=(Button)findViewById(R.id.button);
        predict=(Button)findViewById(R.id.button2);
        tv=(TextView)findViewById(R.id.textView2);
        t1=new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if(status != TextToSpeech.ERROR) {
                    t1.setLanguage(Locale.UK);
                }
            }
        });
        select.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
//                Intent intent=new Intent(Intent.ACTION_GET_CONTENT);
                Intent intent=new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
//                intent.setType("image/*");
                intent.putExtra("android.intent.extra.quickCapture", true);
                startActivityForResult(intent,100);
            }
        });

        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                HashMap<String, Integer> wordToIndices=new HashMap<>();
                HashMap<Integer, String> indexToWords=new HashMap<>();

                String json;
                try {
                    InputStream wi = getAssets().open("sample.json");
                    int size = wi.available();
                    byte[] buffer = new byte[size];
                    wi.read(buffer);
                    wi.close();
                    json = new String(buffer, "UTF-8");
                    Gson gson=new Gson();
                    wordToIndices = gson.fromJson(json, HashMap.class);

                } catch (IOException ex) {
                    ex.printStackTrace();
                }
                try {
                    InputStream iw = getAssets().open("index_to_word.json");
                    int size = iw.available();
                    byte[] buffer = new byte[size];
                    iw.read(buffer);
                    iw.close();
                    json = new String(buffer, "UTF-8");
                    Gson gson=new Gson();
                    indexToWords = gson.fromJson(json, HashMap.class);
//                        tv.setText(indexToWords.get("2341"));
                } catch (IOException ex) {
                    ex.printStackTrace();
                }

                try {
                    interpreter= new Interpreter(loadModelFile(),null);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                Bitmap bitmap = Bitmap.createScaledBitmap(img, 299, 299, true);
                ByteBuffer input = ByteBuffer.allocateDirect(299 * 299 * 3 * 4).order(ByteOrder.nativeOrder());
                ByteBuffer output=ByteBuffer.allocateDirect(1*8*8*2048*4).order(ByteOrder.nativeOrder());

                for (int y = 0; y < 299; y++) {
                    for (int x = 0; x < 299; x++) {
                        int px = bitmap.getPixel(x, y);

                        // Get channel values from the pixel value.
                        int r = Color.red(px);
                        int g = Color.green(px);
                        int b = Color.blue(px);

                        // Normalize channel values to [-1.0, 1.0]. This requirement depends
                        // on the model. For example, some models might require values to be
                        // normalized to the range [0.0, 1.0] instead.
                        float rf = (r - 127) / 255.0f;
                        float gf = (g - 127) / 255.0f;
                        float bf = (b - 127) / 255.0f;

                        input.putFloat(rf);
                        input.putFloat(gf);
                        input.putFloat(bf);
                    }
                }
                input.rewind();

                interpreter.run(input,output);
                output.rewind();

                float[][][] encoutput=new float[1][64][256];
                try {
                    encoder=new Interpreter(loadencoder(),null);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                encoder.resizeInput(0,new int[]{1,64,2048});
                encoder.run(output,encoutput);
                try {
                    decoder=new Interpreter(loaddecoder(),null);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                int[][] decInput=new int[1][1];
                float[][] hidden=new float[1][512];
                decInput[0][0]=3;
                String res="";
                for(int j=0;j<52;j++)
                {
                    Object[] inputs={decInput,encoutput,hidden};
                    float[][] pred=new float[1][5001];
                    float[][][] attn=new float[1][64][1];

                    Map<Integer, Object> outputs = new HashMap<>();
                    outputs.put(0, pred);
                    outputs.put(1, hidden);
                    outputs.put(2, attn);
                    decoder.runForMultipleInputsOutputs(inputs, outputs);
                    float large=0.0f;
                    int predIdx= -1;
                    for(int i=0;i<pred[0].length;i++)
                    {
                        if(pred[0][i]>large){
                            large=pred[0][i];
                            predIdx=i;
                        }
                    }
                    if(predIdx==4)
                        break;
                    decInput[0][0]=predIdx;
                    res+=" "+indexToWords.get(String.valueOf(predIdx));

                }

                tv.setText(res);
                t1.speak(res, TextToSpeech.QUEUE_FLUSH, null);
                interpreter.close();
                encoder.close();
                decoder.close();
            }
        });
    }
    private MappedByteBuffer loadModelFile() throws IOException
    {
        AssetFileDescriptor assetFileDescriptor=this.getAssets().openFd("feature.tflite");
        FileInputStream fileInputStream=new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel=fileInputStream.getChannel();
        long startOffset=assetFileDescriptor.getStartOffset();
        long length=assetFileDescriptor.getLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,length);
    }
    private MappedByteBuffer loadencoder() throws IOException
    {
        AssetFileDescriptor assetFileDescriptor=this.getAssets().openFd("encoder.tflite");
        FileInputStream fileInputStream=new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel=fileInputStream.getChannel();
        long startOffset=assetFileDescriptor.getStartOffset();
        long length=assetFileDescriptor.getLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,length);
    }
    private MappedByteBuffer loaddecoder() throws IOException
    {
        AssetFileDescriptor assetFileDescriptor=this.getAssets().openFd("decoder.tflite");
        FileInputStream fileInputStream=new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel=fileInputStream.getChannel();
        long startOffset=assetFileDescriptor.getStartOffset();
        long length=assetFileDescriptor.getLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,length);
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode==100)
        {
//            imgview.setImageURI(data.getData());
//            Uri uri=data.getData();
//            try {
//                img= MediaStore.Images.Media.getBitmap(this.getContentResolver(),uri);
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
            img= (Bitmap) data.getExtras().get("data");
            imgview.setImageBitmap(img);
        }
    }
}