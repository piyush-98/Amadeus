package com.jskgmail.lifesaver;

import android.annotation.TargetApi;
import android.content.Intent;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

import com.victor.loading.rotate.RotateLoading;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import retrofit2.http.GET;
import ru.dimorinny.floatingtextbutton.FloatingTextButton;

public class MainhospActivity extends AppCompatActivity {
    String TAG = "HOSPt";
    private RotateLoading rotateLoading;
    TextView t,tt,tt1,tt11;

    @TargetApi(Build.VERSION_CODES.O)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ActionBar actionBar = getSupportActionBar();
        actionBar.setHomeButtonEnabled(true);
        actionBar.setDisplayHomeAsUpEnabled(true);
        actionBar.setHomeAsUpIndicator(R.drawable.ic_arrow_back_black_24dp);
        actionBar.setDisplayShowHomeEnabled(true);
        setContentView(R.layout.activity_mainhosp);
        t = (TextView) findViewById(R.id.textView20);
        t.setText("ZIP code : " + MainActivity.ZIP);

        rotateLoading = (RotateLoading) findViewById(R.id.rotateloading);
        rotateLoading.setLoadingColor(R.color.colorPrimary);

        FloatingTextButton call = findViewById(R.id.fab);
        FloatingTextButton dirl = findViewById(R.id.fab1);
        if( MainActivity.hospp.equals("No Hospital Found")) {
            rotateLoading.start();
            goapiihospital();
        }



        call.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String[] em = MainActivity.emergencyno.split(",");
                Uri call = Uri.parse("tel:" + em[0]);
                Intent surf = new Intent(Intent.ACTION_DIAL, call);
                startActivity(surf);
            }
        });

        dirl.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String geoUri = "http://maps.google.com/maps?q=loc:" + MainActivity.latitude;
                Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(geoUri));
                startActivity(intent);
            }
        });
         tt = (TextView) findViewById(R.id.textView35);
        tt.setText(MainActivity.hospp);
         tt1 = (TextView) findViewById(R.id.textView351);
        tt1.setText(MainActivity.hospp1);
         tt11 = (TextView) findViewById(R.id.textView3511);
        tt11.setText(MainActivity.hospp11);

        String b = null;
        try {
            JSONArray obj = new JSONArray(MainActivity.blba);
            for (int i = 0; i < obj.length(); i++) {
                b = b + (obj.getString(i));
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }


    }













    void goapiihospital()
    {




        ApiInterfacehosp apiService = ApiClienthospital.getClient().create(ApiInterfacehosp.class);
//TODO
        Call call = apiService.getall();
        call.enqueue(new Callback() {
            @Override
            public void onResponse(Call call, Response response) {
                Log.e(TAG, "success");
                Log.e(TAG, response.raw().request().url().toString());
                String url = response.raw().request().url().toString();
                Hospitalprocessor mytask = new Hospitalprocessor();
                mytask.execute(url);


            }

            @Override
            public void onFailure(Call call, Throwable t) {
                Log.e(TAG, "failureee");
            }


        });




    }

    private class Hospitalprocessor extends AsyncTask<String, Void, Integer> {



        public Hospitalprocessor() {

            super();

        }


        // The onPreExecute is executed on the main UI thread before background processing is

        // started. In this method, we start the progressdialog.

        @Override

        protected void onPreExecute() {

            super.onPreExecute();


            // Show the progress dialog on the screen


        }


        // This method is executed in the background and will return a result to onPostExecute

        // method. It receives the file name as input parameter.

        @Override

        protected Integer doInBackground(String... urls) {


            InputStream inputStream = null;

            HttpURLConnection urlConnection = null;

            Integer result = 0;


            // TODO connect to server, download and process the JSON string


            // Now we read the file, line by line and construct the

            // Json string from the information read in.

            try {

                /* forming th java.net.URL object */

                URL url = new URL(urls[0]);

                urlConnection = (HttpURLConnection) url.openConnection();



                 /* optional request header */

                urlConnection.setRequestProperty("Content-Type", "application/json");



                /* optional request header */

                urlConnection.setRequestProperty("Accept", "application/json");



                /* for Get request */

                urlConnection.setRequestMethod("GET");

                int statusCode = urlConnection.getResponseCode();



                /* 200 represents HTTP OK */

                if (statusCode == 200) {

                    inputStream = new BufferedInputStream(urlConnection.getInputStream());


                    // Convert the read in information to a Json string

                    String response = convertInputStreamToString(inputStream);


                    // now process the string using the method that we implemented in the previous exercise
                    JSONObject obj=new JSONObject(response);
                    String data=obj.getString("data");
                    String[] dataa=data.split("\\[");
                    for(int i=3;i<1050;i++) {
                        String[] pin = dataa[i].split("\"");
                        String[] pinloc = dataa[i].split(",\"");
                        if(pin[16].replace(",","").equals(MainActivity.ZIP)) {
                            MainActivity.emergencyno=pin[17];
                            MainActivity.hosp=pin[1];
                            MainActivity.hospname=pin[1].replace("\"","");
                            MainActivity.hospp=""+pin[1].replace("\"","");
                            MainActivity.hospp1=""+pin[9].replace("\"","");
                            MainActivity.hospp11=""+MainActivity.emergencyno;



                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    rotateLoading.stop();

                                    t.setText("ZIP code : " + MainActivity.ZIP);
                                    tt.setText(MainActivity.hospp);
                                    tt1.setText(MainActivity.hospp1);
                                    tt11.setText(MainActivity.hospp11);

                                }});
                        }
                    }                        result = 1; // Successful

                } else {

                    result = 0; //"Failed to fetch data!";

                }

            } catch (Exception e) {

                Log.d(TAG, e.getLocalizedMessage());

            }

            return result; //"Failed to fetch data!";

        }

    }


    public static class ApiClienthospital {

        public static final String BASE_URL ="https://data.gov.in/node/356921/datastore/export/";

        private static Retrofit retrofit = null;

        public static Retrofit getClient() {
            if (retrofit==null) {
                retrofit = new Retrofit.Builder()
                        .baseUrl(BASE_URL)
                        .addConverterFactory(GsonConverterFactory.create())
                        .build();
            }
            return retrofit;
        }


    }
    public interface ApiInterfacehosp {

        @GET("json")
        Call<ResponseBody> getall();

    }









    private String convertInputStreamToString(InputStream inputStream) throws IOException {

        BufferedReader bufferedReader = new BufferedReader( new InputStreamReader(inputStream));

        String line = "";

        String result = "";

        while((line = bufferedReader.readLine()) != null){

            result += line;

        }



            /* Close Stream */

        if(null!=inputStream){

            inputStream.close();

        }

        return result;

    }









}
