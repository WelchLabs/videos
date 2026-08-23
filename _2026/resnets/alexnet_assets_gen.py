import numpy as np
import os
from PIL import Image

# data_dir='/Volumes/PG Work/Stephencwelch Dropbox/Pranav Gundu/Welch Labs/videos/_2026/resnets/data'
data_dir='/Users/stephen/videos/_2026/resnets/data'
activations_dir=data_dir+'/activations'
examples_dir=data_dir+'/imagenet_examples'

alexnet=None
tfms=None


def activations(im, feature_cuts, classifier_cuts=()):
    """{'features_2': arr, ..., 'classifier_3': arr, ...} for one PIL image."""
    global alexnet, tfms
    import torch
    if alexnet is None:
        import torchvision.models as models
        from torchvision import transforms
        alexnet=models.alexnet(weights='IMAGENET1K_V1')
        alexnet.eval()
        tfms=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    batch=tfms(im)[None]
    out={}
    with torch.no_grad():
        for cut in feature_cuts:
            out['features_'+str(cut)]=alexnet.features[:cut](batch).cpu().numpy()[0]
        if classifier_cuts:
            pooled=torch.flatten(alexnet.avgpool(alexnet.features(batch)), 1)
            for cut in classifier_cuts:
                out['classifier_'+str(cut)]=alexnet.classifier[:cut](pooled).cpu().numpy()[0]
    return out


def save(tag, acts, im):
    out=activations_dir+'/'+tag
    os.makedirs(out, exist_ok=True)
    for name, arr in acts.items():
        np.save(out+'/'+name+'.npy', arr.astype(np.float32))
    small=np.array(im.resize((128, 128)))
    np.save(out+'/im_numpy.npy', small)
    #The scenes texture a quad with this; green stands in for blue, as in the original
    Image.fromarray(small[:,:,[0,1,1]], 'RGB').save(out+'/im.png')


def main():
    im=Image.open(data_dir+'/hot_dog.png').convert('RGB') #p21-p36 and p43
    save('hot_dog', activations(im, [2, 3, 5, 8, 10, 12], [3, 6, 7]), im)
    print('activations/hot_dog')

    #conv-1 kernel weights, for p24b / p24c / p24d
    w=alexnet.features[0].weight.detach().cpu().numpy()
    os.makedirs(activations_dir+'/weights', exist_ok=True)
    np.save(activations_dir+'/weights/features0.npy', w)
    print('activations/weights/features0.npy', w.shape)

    paths=sorted(examples_dir+'/'+f for f in os.listdir(examples_dir)
                 if f.lower().endswith(('.jpeg', '.jpg', '.png')))
    for path in paths: #The p45 series
        stem=os.path.splitext(os.path.basename(path))[0]
        im=Image.open(path).convert('RGB')
        save('imagenet/'+stem, activations(im, [2, 5, 8, 10, 12], [3, 6, 7]), im)
    print('activations/imagenet/', len(paths), 'images')


if __name__=='__main__':
    main()
