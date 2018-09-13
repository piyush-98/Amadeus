package com.jskgmail.lifesaver;

        import android.content.Intent;
        import android.net.Uri;
        import android.os.AsyncTask;
        import android.os.Bundle;
        import android.support.v7.app.ActionBar;
        import android.support.v7.app.AppCompatActivity;
        import android.util.Log;
        import android.view.View;
        import android.widget.TextView;

        import com.victor.loading.rotate.RotateLoading;

        import org.json.JSONArray;
        import org.json.JSONException;

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

public class MainbbActivity extends AppCompatActivity {
    private RotateLoading rotateLoading;
TextView t,tt,tt1,tt11;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ActionBar actionBar = getSupportActionBar();
        actionBar.setHomeButtonEnabled(true);
        actionBar.setDisplayHomeAsUpEnabled(true);
        actionBar.setHomeAsUpIndicator(R.drawable.ic_arrow_back_black_24dp);
        actionBar.setDisplayShowHomeEnabled(true);
        setContentView(R.layout.activity_mainbb);
         t = (TextView) findViewById(R.id.textView20);
        t.setText("ZIP code : " + MainActivity.ZIP);

        rotateLoading = (RotateLoading) findViewById(R.id.rotateloading);
        rotateLoading.setLoadingColor(R.color.colorPrimary);

        FloatingTextButton call = findViewById(R.id.fab);
        FloatingTextButton dirl = findViewById(R.id.fab1);


            if (MainActivity.blba.equals("No BloodBank Found")) {
                rotateLoading.start();
goapibloodbank();
            }

        call.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String[] em = MainActivity.bloodno.split(",");
                Uri call = Uri.parse("tel:" + em[0]);
                Intent surf = new Intent(Intent.ACTION_DIAL, call);
                startActivity(surf);

            }
        });
        dirl.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String geoUri = "http://maps.google.com/maps?q=loc:" + MainActivity.bloodloc;
                Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(geoUri));
                startActivity(intent);
            }
        });
         tt = (TextView) findViewById(R.id.textView35);
        tt.setText(MainActivity.blba);
         tt1 = (TextView) findViewById(R.id.textView351);
        tt1.setText(MainActivity.blba1);
         tt11 = (TextView) findViewById(R.id.textView3511);
        tt11.setText(MainActivity.blba11);
        Log.e("bllba", MainActivity.blba);
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



    public static class ApiClientblood{

        public static final String BASE_URL ="https://data.gov.in/node/3287321/datastore/export/";

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
    public interface ApiInterfaceblood{

        @GET("json")
        Call<ResponseBody> getall();

    }


    void goapibloodbank()
    {
        ApiInterfaceblood apiService = ApiClientblood.getClient().create(ApiInterfaceblood.class);
//TODO
        Call call = apiService.getall();
        call.enqueue(new Callback() {
            @Override
            public void onResponse(Call call, Response response) {

                String url = response.raw().request().url().toString();
                Log.e("blll", url);
                BloodB mytask = new BloodB();
                mytask.execute(url);


            }

            @Override
            public void onFailure(Call call, Throwable t) {

            }


        });

    }

    private class BloodB extends AsyncTask<String, Void, Integer> {



        public BloodB() {

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
                    String[] res=response.split("],");
                    for(int i=1;i<res.length;i++)
                    {String[] r=res[i].split(",\"");

                        if(r[6].replace("\"","").equals(MainActivity.ZIP)) {

                            MainActivity.bloodno=r[7].replace("\"","");
                            MainActivity.bloodloc=r[r.length-2].replace("\"","")+","+r[r.length-1].replace("\"","");

                            MainActivity.bbname=r[1].replace("\"","");
                            MainActivity.blba=""+r[1].replace("\"","");
                            MainActivity.blba1=""+r[5].replace("\"","");
                            MainActivity.blba11=""+r[7].replace("\"","");
                        }
                    }

/*
                    String ele = (obj.getJSONArray("results")).toString();
                    JSONObject obj1=new JSONObject(ele.replaceFirst("\\[",""));
                    String eee=obj1.getJSONArray("address_components").toString();
                    Log.e("resppp",eee);
                    JSONArray arr=obj1.getJSONArray("address_components");
                    for(int i=0;i<arr.length();i++){
                        if(arr.getString(i).contains("postal_code"))
                            Log.e("respppp", arr.getString(i));
                        JSONObject obj11=new JSONObject(arr.getString(i));
                        Log.e("resppppp", obj11.getString("long_name"));
                        ZIP=obj11.getString("long_name");
                        goapiihospital();


                    }*/

                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            rotateLoading.stop();
                            t.setText("ZIP code : " + MainActivity.ZIP);
                            tt.setText(MainActivity.blba);
                            tt1.setText(MainActivity.blba1);
                            tt11.setText(MainActivity.blba11);
                        }});
                    result = 1; // Successful

                } else {

                    result = 0; //"Failed to fetch data!";

                }

            } catch (Exception e) {


            }

            return result; //"Failed to fetch data!";

        }

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


