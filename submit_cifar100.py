from diffusers import AutoPipelineForText2Image
import torch
from tqdm import tqdm
import albumentations as A
from PIL import Image
import numpy as np
import ipdb
import argparse
import os
import sys
import time
import random
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar100', type=str, 
                        help='data prepare to distillate:imagenet/tiny-imagenet')
    parser.add_argument('--ipc', default=100, type=int, 
                        help='image per class')
    parser.add_argument('--aug_n', default=4, type=int, 
                        help='aug per image')
    parser.add_argument('--size', default=32, type=int, 
                        help='init resolution (resize)')
    parser.add_argument('--save_image_path', default='./', type=str, 
                        help='where to save the generated files')
                        
    args = parser.parse_args()
    return args

def augment_image(pil_image, size):
    transform = A.Compose([
        A.RandomCrop(width=size, height=size),
        A.HorizontalFlip(p=0.8),
        A.VerticalFlip(p=0.8),
        A.RandomBrightnessContrast(p=0.5),
        A.Rotate(limit=60, p=0.8),
        A.RandomGamma(p=0.5)
    ])

    image_np = np.array(pil_image)
    augmented_image_np = transform(image=image_np)['image']
    augmented_image = Image.fromarray(augmented_image_np)

    return augmented_image


import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

def gen_syn_images(pipe, label_list, args):
    image_tensors = []
    
    mean = [0.5071, 0.4866, 0.4409]
    std = [0.2673, 0.2564, 0.2762]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    for prompt in tqdm(label_list, total=len(label_list), position=0):
        for i in range(int(args.ipc/5)):
            images = pipe(prompt=prompt, negative_prompt='cartoon, anime, painting', num_inference_steps=1, guidance_scale=0.0).images
            
            # 保存原始图片
            resized_image = images[0].resize((args.size, args.size))
            image_tensor = transform(resized_image)
            image_tensors.append(image_tensor)
            
            for j in range(args.aug_n):
                aug = augment_image(resized_image, size=args.size)
                aug_tensor = transform(aug)
                image_tensors.append(aug_tensor)

    # 将所有图片张量转换为一个四维张量 (N, C, H, W)
    images_tensor = torch.stack(image_tensors)
    
    # 保存为 .pt 文件
    torch.save(images_tensor, 'cifar100.pt')
    print("All images have been saved in cifar100.pt")



def gen_label_list(args):
    if args.dataset == 'tiny-imagenet':
        labels = ['"goldfish,Carassiusauratus"', '"Europeanfiresalamander,Salamandrasalamandra"', '"bullfrog,Ranacatesbeiana"', '"tailedfrog,belltoad,ribbedtoad,tailedtoad,Ascaphustrui"', '"Americanalligator,Alligatormississipiensis"', '"boaconstrictor,Constrictorconstrictor"', 'trilobite', 'scorpion', '"blackwidow,Latrodectusmactans"', 'tarantula', 'centipede', 'goose', '"koala,koalabear,kangaroobear,nativebear,Phascolarctoscinereus"', 'jellyfish', 'braincoral', 'snail', 'slug', '"seaslug,nudibranch"', '"Americanlobster,Northernlobster,Mainelobster,Homarusamericanus"', '"spinylobster,langouste,rocklobster,crawfish,crayfish,seacrawfish"', '"blackstork,Ciconianigra"', '"kingpenguin,Aptenodytespatagonica"', '"albatross,mollymawk"', '"dugong,Dugongdugon"', 'Chihuahua', 'Yorkshireterrier', 'goldenretriever', 'Labradorretriever', '"Germanshepherd,Germanshepherddog,Germanpolicedog,alsatian"', 'standardpoodle', '"tabby,tabbycat"', 'Persiancat', 'Egyptiancat', '"cougar,puma,catamount,mountainlion,painter,panther,Felisconcolor"', '"lion,kingofbeasts,Pantheraleo"', '"brownbear,bruin,Ursusarctos"', '"ladybug,ladybeetle,ladybeetle,ladybird,ladybirdbeetle"', 'fly', 'bee', '"grasshopper,hopper"', '"walkingstick,walkingstick,stickinsect"', '"cockroach,roach"', '"mantis,mantid"', '"dragonfly,darningneedle,devil\'sdarningneedle,sewingneedle,snakefeeder,snakedoctor,mosquitohawk,skeeterhawk"', '"monarch,monarchbutterfly,milkweedbutterfly,Danausplexippus"', '"sulphurbutterfly,sulfurbutterfly"', '"seacucumber,holothurian"', '"guineapig,Caviacobaya"', '"hog,pig,grunter,squealer,Susscrofa"', 'ox', 'bison', '"bighorn,bighornsheep,cimarron,RockyMountainbighorn,RockyMountainsheep,Oviscanadensis"', 'gazelle', '"Arabiancamel,dromedary,Camelusdromedarius"', '"orangutan,orang,orangutang,Pongopygmaeus"', '"chimpanzee,chimp,Pantroglodytes"', 'baboon', '"Africanelephant,Loxodontaafricana"', '"lesserpanda,redpanda,panda,bearcat,catbear,Ailurusfulgens"', 'abacus', '"academicgown,academicrobe,judge\'srobe"', 'altar', 'apron', '"backpack,backpack,knapsack,packsack,rucksack,haversack"', '"bannister,banister,balustrade,balusters,handrail"', 'barbershop', 'barn', '"barrel,cask"', 'basketball', '"bathtub,bathingtub,bath,tub"', '"beachwagon,stationwagon,wagon,estatecar,beachwaggon,stationwaggon,waggon"', '"beacon,lighthouse,beaconlight,pharos"', 'beaker', 'beerbottle', '"bikini,two-piece"', '"binoculars,fieldglasses,operaglasses"', 'birdhouse', '"bowtie,bow-tie,bowtie"', '"brass,memorialtablet,plaque"', 'broom', '"bucket,pail"', '"bullettrain,bullet"', '"butchershop,meatmarket"', '"candle,taper,waxlight"', 'cannon', 'cardigan', '"cashmachine,cashdispenser,automatedtellermachine,automatictellermachine,automatedteller,automaticteller,ATM"', 'CDplayer', 'chain', 'chest', 'Christmasstocking', 'cliffdwelling', '"computerkeyboard,keypad"', '"confectionery,confectionary,candystore"', 'convertible', 'crane', '"dam,dike,dyke"', 'desk', '"diningtable,board"', 'drumstick', 'dumbbell', '"flagpole,flagstaff"', 'fountain', 'freightcar', '"fryingpan,frypan,skillet"', 'furcoat', '"gasmask,respirator,gashelmet"', 'go-kart', 'gondola', 'hourglass', 'iPod', '"jinrikisha,ricksha,rickshaw"', 'kimono', '"lampshade,lampshade"', '"lawnmower,mower"', 'lifeboat', '"limousine,limo"', 'magneticcompass', 'maypole', 'militaryuniform', '"miniskirt,mini"', 'movingvan', 'nail', 'neckbrace', 'obelisk', '"oboe,hautboy,hautbois"', '"organ,pipeorgan"', 'parkingmeter', '"pay-phone,pay-station"', '"picketfence,paling"', 'pillbottle', '"plunger,plumber\'shelper"', 'pole', '"policevan,policewagon,paddywagon,patrolwagon,wagon,blackMaria"', 'poncho', '"popbottle,sodabottle"', "potter'swheel", '"projectile,missile"', '"punchingbag,punchbag,punchingball,punchball"', 'reel', '"refrigerator,icebox"', '"remotecontrol,remote"', '"rockingchair,rocker"', 'rugbyball', 'sandal', 'schoolbus', 'scoreboard', 'sewingmachine', 'snorkel', 'sock', 'sombrero', 'spaceheater', '"spiderweb,spider\'sweb"', '"sportscar,sportcar"', 'steelarchbridge', '"stopwatch,stopwatch"', '"sunglasses,darkglasses,shades"', 'suspensionbridge', '"swimmingtrunks,bathingtrunks"', 'syringe', 'teapot', '"teddy,teddybear"', '"thatch,thatchedroof"', 'torch', 'tractor', 'triumphalarch', '"trolleybus,trolleycoach,tracklesstrolley"', 'turnstile', 'umbrella', 'vestment', 'viaduct', 'volleyball', 'waterjug', 'watertower', 'wok', 'woodenspoon', 'comicbook', 'plate', 'guacamole', '"icecream,icecream"', '"icelolly,lolly,lollipop,popsicle"', 'pretzel', 'mashedpotato', 'cauliflower', 'bellpepper', 'mushroom', 'orange', 'lemon', 'banana', 'pomegranate', '"meatloaf,meatloaf"', '"pizza,pizzapie"', 'potpie', 'espresso', 'alp', '"cliff,drop,drop-off"', 'coralreef', '"lakeside,lakeshore"', '"seashore,coast,seacoast,sea-coast"', 'acorn']
    elif args.dataset == 'cifar100':
        labels = ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium fish', 'flatfish', 'ray', 'shark', 'trout', 'orchids', 'poppies', 'roses', 'sunflowers', 'tulips', 'bottles', 'bowls', 'cans', 'cups', 'plates', 'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers', 'clock', 'computer keyboard', 'lamp', 'telephone', 'television', 'bed', 'chair', 'couch', 'table', 'wardrobe', 'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 'bear', 'leopard', 'lion', 'tiger', 'wolf', 'bridge', 'castle', 'house', 'road', 'skyscraper', 'cloud', 'forest', 'mountain', 'plain', 'sea', 'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', 'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'crab', 'lobster', 'snail', 'spider', 'worm', 'baby', 'boy', 'girl', 'man', 'woman', 'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle', 'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', 'maple', 'oak', 'palm', 'pine', 'willow', 'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train', 'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor']
    return labels

def main():
    args = parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    label_dic = gen_label_list(args)
    pipe = AutoPipelineForText2Image.from_pretrained("./", torch_dtype=torch.float16, variant="fp16")
    # pipe = AutoPipelineForText2Image.from_pretrained("./")
    pipe.to("cuda")
    gen_syn_images(pipe=pipe, label_list=label_dic, args=args)


if __name__ == "__main__" : 
    main()
