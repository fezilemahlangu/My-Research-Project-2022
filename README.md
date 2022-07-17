# My-Research-Project-2022
    Title: Surrogate Tasks for Architecture Selection in Reinforcement Learning

## Meta data description 

    meta_data.csv: meta data for the dataset., including:

    GameName: String. Game name. e.g., “alien” indicates the trial is collected for game Alien (15 min time limit). “alien_highscore” is the trajectory collected from the best player’s highest score (2 hour limit). See dataset description paper for details.

    trial_id: Integer. One can use this number to locate the associated .tar.bz2 file and label file.

    subject_id: Char. Human subject identifiers.

    load_trial: Integer. 0 indicates that the game starts from scratch. If this field is non-zero, it means that the current trial continues from a saved trial. The number indicates the trial number to look for.

    highest_score: Integer. The highest game score obtained from this trial.

    total_frame: Number of image frames in the .tar.bz2 repository.

    total_game_play_time: Integer. game time in ms. 

    total_episode: Integer. number of episodes in the current trial. An episode terminates when all lives are consumed.

    avg_error: Float. Average eye-tracking validation error at the end of each trial in visual degree (1 visual degree = 1.44 cm in our experiment). See our paper for the calibration/validation process.

    max_error: Float. Max eye-tracking validation error. 

    low_sample_rate: Percentage. Percentage of frames with less than 10 gaze samples. The most common reason for this is blinking.

    frame_averaging: Boolean. The game engine allows one to turn this on or off. When turning on (TRUE), two consecutive frames are averaged, this alleviates screen flickering in some games.

    fps: Integer. Frame per second when an action key is held down.
