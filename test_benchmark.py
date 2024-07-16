import os.path
import logging
import time
from collections import OrderedDict
import torch

from util import utils_logger
from util import utils_image as util
from basicsr.archs.RFDN_arch import RFDN
from basicsr.archs.BSRN_arch import BSRN
from torchstat import stat
from calflops import calculate_flops

def main():

    utils_logger.logger_info('AIM-track', log_path='AIM-track.log')
    logger = logging.getLogger('AIM-track')

    # --------------------------------
    # basic settings
    # --------------------------------
    testsets = 'Manga109'
    #testset_L = 'DIV2K_valid_LR_bicubic'
    #testset_L = 'DIV2K_test_LR_bicubic'

    torch.cuda.current_device()
    torch.cuda.empty_cache()
    #torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # --------------------------------
    # read image
    # --------------------------------
    #L_folder = os.path.join(testsets, testset_L, 'X4')
    #E_folder = os.path.join(testsets, testset_L+'_results')

    H_folder = 'E:\\SISR\\datasets\\Input\\Set14\\GTmod4'
    L_folder = 'E:\SISR\SRCNN\Flickr2K\Flickr2K_test_LR_bicubic\X4'
    E_folder = 'E:\\SISR\\SRCNN\\PCDB-main\\Output\\Set14\\SRbicx4'
    util.mkdir(E_folder)

    # --------------------------------
    # load model
    # --------------------------------
    #model_path = os.path.join('trained_model', 'RFDN_AIM.pth')
    model_path = r'E:\SISR\SRCNN\PCDB-main\experiments\ablation_RFDN_pw\models\net_g_486000.pth'
    #model = BSRN(3,64,8,3,4,'BSConvU')
    model = RFDN()
    
    model.load_state_dict(torch.load(model_path)['params'], strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    print(stat(model, (3, 320, 180)))
    exit()
    
    input_shape = (1, 3, 320, 180)
    flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)
    print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))  
    exit()
    
    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))
    
    # record PSNR, runtime
    test_results = OrderedDict()
    test_results['runtime'] = []

    #logger.info(L_folder)
    #logger.info(E_folder)
    idx = 0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    img_SR = []
    for img in util.get_image_paths(L_folder):

        # --------------------------------
        # (1) img_L
        # --------------------------------
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info('{:->4d}--> {:>10s}'.format(idx, img_name+ext))

        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)

        start.record()
        img_E = model(img_L)
        end.record()
        torch.cuda.synchronize()
        test_results['runtime'].append(start.elapsed_time(end))  # milliseconds

        # --------------------------------
        # (2) img_E
        # --------------------------------
        #img_E = util.tensor2uint(img_E)
        #img_SR.append(img_E)

        # --------------------------------
        # (3) save results
        # --------------------------------
        #util.imsave(img_E, os.path.join(E_folder, img_name+ext))

    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    logger.info('------> Average runtime of ({}) is : {:.6f} seconds'.format(L_folder, ave_runtime))

    # --------------------------------
    # (4) calculate psnr
    # --------------------------------
    
    psnr = []
    idx = 0
    '''
    for img in util.get_image_paths(H_folder):
        img_H = util.imread_uint(img, n_channels=3)
        psnr.append(util.calculate_psnr(img_SR[idx], img_H))
        idx += 1
    logger.info('------> Average psnr of ({}) is : {:.6f} dB'.format(L_folder, sum(psnr)/len(psnr)))
    '''

if __name__ == '__main__':

    main()
