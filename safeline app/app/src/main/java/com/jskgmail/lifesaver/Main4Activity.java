package com.jskgmail.lifesaver;

import android.os.Bundle;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;

import com.pierfrancescosoffritti.youtubeplayer.player.AbstractYouTubePlayerListener;
import com.pierfrancescosoffritti.youtubeplayer.player.YouTubePlayer;
import com.pierfrancescosoffritti.youtubeplayer.player.YouTubePlayerInitListener;
import com.pierfrancescosoffritti.youtubeplayer.player.YouTubePlayerView;

public class Main4Activity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        ActionBar actionBar=getSupportActionBar();
        actionBar.setHomeButtonEnabled(true);
        actionBar.setDisplayHomeAsUpEnabled(true);

        actionBar.setHomeAsUpIndicator(R.drawable.ic_arrow_back_black_24dp);
        actionBar.setDisplayShowHomeEnabled(true);
        setContentView(R.layout.activity_main4);
       go1();
    }
void go1()
{
    YouTubePlayerView youTubePlayerView = findViewById(R.id.youtube_player_view);

    youTubePlayerView.initialize(new YouTubePlayerInitListener() {
        @Override
        public void onInitSuccess(final YouTubePlayer initializedYouTubePlayer) {
            initializedYouTubePlayer.addListener(new AbstractYouTubePlayerListener() {
                @Override
                public void onReady() {
                    String videoId = "43M5mZuzHF8";
                    initializedYouTubePlayer.loadVideo(videoId, 0);
                }
            });
        }
    }, true);



}


    private void go() {
/*VideoView v=(VideoView)findViewById(R.id.videoView) ;
        final ProgressBar p=(ProgressBar)findViewById(R.id.progressBar3);
        v.setOnPreparedListener(new MediaPlayer.OnPreparedListener() {
            @Override
            public void onPrepared(MediaPlayer mp) {
                p.setVisibility(View.GONE);
            }
        });
        String s="https://firebasestorage.googleapis.com/v0/b/lifesaver-18f28.appspot.com/o/flood.mp4?alt=media&token=179d7e4e-7171-4a87-b1f8-b1fc3d976c60";
        Uri uri=Uri.parse(s);
        v.setVideoURI(uri);
        v.setVideoPath(s);
        v.requestFocus();
        v.start();
*/

    }









}
