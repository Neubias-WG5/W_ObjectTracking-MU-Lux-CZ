import sys
import os
import numpy as np
from skimage import io
from cell_tracker import process_dataset 
from cytomine.models import Job
from neubiaswg5 import CLASS_OBJTRK
from neubiaswg5.helpers import NeubiasJob, prepare_data, upload_data, upload_metrics


def main(argv):
    # 0. Initialize Cytomine client and job if necessary and parse inputs
    with NeubiasJob.from_cli(argv) as nj:
        problem_cls = get_discipline(nj, default=CLASS_OBJTRK)
        is_2d = False
        nj.job.update(status=Job.RUNNING, progress=0,
                      statusComment="Running workflow for problem class '{}'".format(problem_cls))

        # 1. Prepare data for workflow
        in_imgs, gt_imgs, in_path, gt_path, out_path, tmp_path = prepare_data(problem_cls, nj, **nj.flags)

        # 2. Run image analysis workflow
        nj.job.update(progress=25, statusComment="Launching workflow...")

        for in_img in in_imgs:

            # convert the image data to Cell Tracking Challenge format
            img = io.imread(os.path.join(in_path, in_img))
            T = img.shape[0]
            Y = img.shape[1]
            X = img.shape[2]
            img_data = img.ravel()
            index = 0
            offset = Y*X
            for t in range(T):
                io.imsave(os.path.join(out_dir, template + '{0:03d}.tif'.format(t)),img_data[index:index+offset].reshape((Y,X)))
                index += offset

            # do segmentation and tracking
            process_dataset(tmp_path, tmp_path, '/app/model.h5')

            # convert the tracking results to the required format
            index = 0
            res_img = np.zeros((T,Y,X),np.uint16)
            res_data = res_img.ravel()
            for t in range(T):
                res = io.imread(os.path.join(out_dir, 'mask{0:03d}.tif'.format(t)))
                res_data[index:index+offset]=res.ravel()
                index += offset
            io.imsave(os.path.join(out_path, in_img), res_img)
            os.rename(os.path.join(tmp_path, 'res_track.txt'), os.path.join(out_path, img.filename_no_extension+'.txt')

        # 4. Upload the annotation and labels to Cytomine
        upload_data(problem_cls, nj, in_imgs, out_path, **nj.flags, is_2d=is_2d, monitor_params={"start": 60, "end": 90, "period": 0.1})

        # 5. Compute and upload the metrics
        nj.job.update(progress=90, statusComment="Computing and uploading metrics...")
        upload_metrics(problem_cls, nj, in_imgs, gt_path, out_path, tmp_path, **nj.flags)

        # 6. End
        nj.job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.")


if __name__ == "__main__":
    main(sys.argv[1:])
