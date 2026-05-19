class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = '/media/user/新加卷/data/UCF-101'

            # Save preprocess data into output_dir
            # output_dir = '/media/user/新加卷/fuyu/videoMamba-ucf101/ucf'
            output_dir = '/media/user/新加卷/fuyu/pytorch-video-recognition-master/data'

            return root_dir, output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = '/media/user/新加卷/data/hmdb51'

            output_dir = '/media/user/新加卷/fuyu/pytorch-video-recognition-master/hmdb'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return '/media/user/新加卷/fuyu/pytorch-video-recognition-master/pretrained/ucf101-caffe.pth'