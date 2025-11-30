import os
import cv2
import shutil
import deeplabcut

file = '/Users/acalapai/Desktop/Collage/multiview/20250711_multiview_preview.mp4'
full_video_path = '/Users/acalapai/Desktop/Collage/multiview/20250711_multiview_preview.mp4'
videotype = os.path.splitext(file)[-1].lstrip('.')  # or MOV, or avi, whatever you uploaded!
video_down = deeplabcut.DownSampleVideo(file, width=900)

model_options = deeplabcut.create_project.modelzoo.Modeloptions
model_selection = 'full_macaque'

project_name = 'DLC_FullBody-v2'
your_name = 'anc'

config_path, train_config_path = deeplabcut.create_pretrained_project(
    project_name,
    your_name,
    [video_down],
    videotype=videotype,
    model=model_selection,
    analyzevideo=True,
    createlabeledvideo=False,
    copy_videos=True,
)

edits = {
    'dotsize': 3,  # size of the dots!
    'pcutoff': 0.1,  # the higher, the more conservative the plotting!
}
deeplabcut.auxiliaryfunctions.edit_config(config_path, edits)

project_path = os.path.dirname(config_path)
full_video_path = os.path.join(
    project_path,
    'videos',
    os.path.basename(video_down),
)

# filter predictions (should already be done above ;):
deeplabcut.filterpredictions(config_path, [full_video_path], videotype=videotype)

# re-create the video with your edits!
deeplabcut.create_labeled_video(config_path, [full_video_path], videotype=videotype, filtered=True)

