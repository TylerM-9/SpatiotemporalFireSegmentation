
class Path(object):
    @staticmethod
    def db_root_dir():
        return '/home/c43n256/Data/DAVIS'

    @staticmethod
    def save_root_dir():
        return '/home/group/farazdadgostari/Testing/runs'

    @staticmethod
    def models_dir():
        return "./models"

    @staticmethod
    def data_dir():
        return "./data"

    @staticmethod
    def VID_list_file():
        return "/home/c43n256/REU2026/SpatiotemporalFireSegmentation/stcnn/data/VID_seqs_list.txt"

    @staticmethod
    def DAVIS_list_file():
        return "./data/DAVIS_seqs_list.txt"

    @staticmethod
    def MSRAdataset_dir():
        return "/home/c43n256/Data/MSRA10K_Imgs_GT/Imgs/"

    @staticmethod
    def VOC_dir():
        return "/home/c43n256/Data/"

    @staticmethod
    def DAVIS16_samples_list():
        return "/home/c43n256/REU2026/SpatiotemporalFireSegmentation/stcnn/data/DAVIS16_samples_list.txt"

    # pretrained_netG_dict = torch.load(os.path.join('/home/c43n256/Data', 'NetG_epoch-99.pth'), map_location=device)
    @staticmethod
    def pretrained_netG_dict():
        return "/home/c43n256/Data/NetG_epoch-99.pth"

    # initialize_netD(netD, os.path.join('/home/c43n256/Data', 'NetD_epoch-99.pth'))
    @staticmethod
    def pretrained_netD_dict():
        return "/home/c43n256/Data/NetD_epoch-99.pth"

    @staticmethod
    def default_output():
        return "/home/c43n256/REU2026/SpatiotemporalFireSegmentation/stcnn/output"

    # def __init__(self, train=True, inputRes=None, db_root_dir='/home/c43n256/Dataset/DAVIS/', transform=None, seq_name=None, frame_nums=4):
    @staticmethod
    def davis_db_root_dir():
        return '/home/c43n256/Dataset/DAVIS/'

    # NOTE: These two paths are used for the archive-2 dataset and not needed for the current project
    # def __init__(self, inputRes=None, image_path="/home/c43n256/Data/archive-2/Image/Merged", mask_path="/home/c43n256/Data/archive-2/Mask/Fire", transform=None, num_frame=4):
    @staticmethod
    def archive2_image_path():
        return "/home/c43n256/Data/archive-2/Image/Merged"

    @staticmethod
    def archive2_mask_path():
        return "/home/c43n256/Data/archive-2/Mask/Fire"

    # def __init__(self, inputRes=None, samples_path="/home/c43n256/Data/Mask_Data", transform=None, mode="train"):
    @staticmethod
    def flame_mask_samples_path():
        return "/home/c43n256/Data/Mask_Data"