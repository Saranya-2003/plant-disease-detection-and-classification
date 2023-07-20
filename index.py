from email import header
import streamlit as st
import torch
from torch import nn
from PIL import Image
import base64
import torchvision.transforms as transforms

# for moving data into GPU (if available)
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available:
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# for moving data to device (CPU or GPU)
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

device = get_default_device()

# for calculating the accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# base class for the model
class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
  
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                   # Generate prediction
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)          # Calculate accuracy
        return {"val_loss": loss.detach(), "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()       # Combine loss  
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy} # Combine accuracies
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))

# Architecture for training
# convolution block with BatchNormalization
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

# resnet architecture 
class ResNet9(ImageClassificationBase):
    def init(self, in_channels, num_diseases):
        super().init()
 
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) # out_dim : 128 x 64 x 64 
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True) # out_dim : 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True) # out_dim : 512 x 4 x 44
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))
        
    def forward(self, xb): # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out     

classes = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)__Common_rust',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape__Esca(Black_Measles)',
 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange__Haunglongbing(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,bell__Bacterial_spot',
 'Pepper,bell__healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

disease_info = {
    'apple scab':{
        'description': 'A serious disease of apples and ornamental crabapples, apple (scab Venturia inaequalis) attacks both leaves and fruitIt produces lesions that can be observed on leaves, as well as blossoms, sepals, pedicels, petioles, and even the fruit itselfThe fungus called Venturia inaequalis is what causes apple scab. It can infect apples and crab apples (Malus spp.), mountain ash (Sorbus spp.), hawthorn (Crataegus spp.), mountain ash (Sorbus spp.), loquat (Eriobotrya japonica), and firethorn (Pyracantha spp.)',
        'symptoms': ['The first signs of apple scab depend on the type of tree being infected, but generally initial symptoms include leaves turning dull, with olive-green round spots, usually on the leaves closest to the buds.'],
        'diagnosis':['Managing and treating the apple scab fungus is an integrated process that combines sanitation, resistant cultivars, and fungicides ',
'•	Choose Scab-Resistant Cultivars',
'•	Plant Trees Correctly',
'•	Prune Infected Leaves',
'•	Change Watering Schedule',
'•	Cover with Compost',
'•	Spray Liquid Copper Soap',
'•	Use Wettable Sulfur',
'•	Apply Fungicides',
'•	Use Sulfur Based Sprays'
        ],
        'video_url':'https://www.youtube.com/watch?v=SMS2jHqACrY&pp=ygUWaG93IHRvIGN1cmUgYXBwbGUgc2NhYg%3D%3D'
    },
    'black rot': {
        'description': 'Large brown rotten areas can form anywhere on the fruit but are most common on the blossom end.Brown to black concentric rings can often be seen on larger infections.The flesh of the apple is brown but remains firm.',
        'symptoms': ['Leaf symptoms-Infected leaves develop "frog-eye leaf spot”. The se are circular spots with purplish or reddish edges and light tan interiors.', 'Branch symptoms-Cankers appear as a sunken, reddish-brown area on infected branches. Cankers often have rough or cracked bark.Cankers may be hard to see. If you see rotten fruit or frog-eye leaf spots, inspect your trees for cankers.'],
        'diagnosis':[
'•	Prune out dead or diseased branches.',
'•	Pick all dried and shriveled fruits remaining on the trees.',
'•	Remove infected plant material from the area.',
'•	All infected plant parts should be burned, buried or sent to a municipal composting site.',
'•	Be sure to remove the stumps of any apple trees you cut down. Dead stumps can be a source of spores.'
],
        'video_url':'https://www.youtube.com/watch?v=HT_qdcl0cp8&pp=ygUbaG93IHRvIGN1cmUgYXBwbGUgYmxhY2sgcm90'
    },
    'cedar apple rust': {
        'description':'Cedar apple rust (Gymnosporangium juniperi-virginianae) is a fungal disease that requires juniper plants to complete its complicated two year life-cycle. Spores overwinter as a reddish-brown gall on young twigs of various juniper species. In early spring, during wet weather, these galls swell and bright orange masses of spores are blown by the wind where they infect susceptible apple and crab-apple trees.',
        'symptoms': ['Symptoms include wilting, stunted foliage, chlorosis and eventual death of seedlings especially in low, wet areas. Infected seedlings have cambium which is red-brown or butterscotch-colored particularly at the root collar. Feeder roots are dead, black and fine. Trees with root rot turn yellow-red and die, sometimes suddenly.'],
        'diagnosis':['•	Choose resistant cultivars when available.',
'•	Rake up and dispose of fallen leaves and other debris from under trees.',
'•	Remove galls from infected junipers. In some cases, juniper plants should be removed entirely.'
'•	Apply preventative, disease-fighting fungicides labeled for use on apples weekly, starting with bud break, to protect trees from spores being released by the juniper host. This occurs only once per year, so additional applications after this springtime spread are not necessary.',
'•	On juniper, rust can be controlled by spraying plants with a copper solution (0.5 to 2.0 oz/ gallon of water) at least four times between late August and late October.',
'•	Safely treat most fungal and bacterial diseases with SERENADE Garden. This broad spectrum bio-fungicide uses a patented strain of Bacillus subtilis that is registered for organic use. Best of all, SERENADE is completely non-toxic to honey bees and beneficial insects.',
'•	Containing sulfur and pyrethrins, Bonide® Orchard Spray is a safe, one-hit concentrate for insect attacks and fungal problems. For best results, apply as a protective spray (2.5 oz/ gallon) early in the season. If disease, insects or wet weather are present, mix 5 oz in one gallon of water. Thoroughly spray all parts of the plant, especially new shoots.',
'•	Contact your local Agricultural Extension office for other possible solutions in your area.'],
        'video_url' : 'https://www.youtube.com/watch?v=K9M8CkaIQho&pp=ygUZY2VkYXIgYXBwbGUgcm90IHRyZWF0bWVudA%3D%3D'
    },
    'healthy':{
    'description': 'This is a healthy green leaf.It does not contain any diseases',
    },
    'healthy':{
        'description': 'This is a healthy green leaf.It does not contain any diseases',
    },
    '(including sour) powdery mildew':{
        'description':'Powdery mildew of sweet and sour cherry is caused by Podosphaera clandestina, an obligate biotrophic fungus. Mid- and late-season sweet cherry (Prunus avium) cultivars are commonly affected, rendering them unmarketable due to the covering of white fungal growth on the cherry surface .Season long disease control of both leaves and fruit is critical to minimize overall disease pressure in the orchard and consequently to protect developing fruit from accumulating spores on their surfaces.',
        'symptoms':['Initial symptoms, often occurring 7 to 10 days after the onset of the first irrigation, are light roughly-circular, powdery looking patches on young, susceptible leaves (newly unfolded, and light green expanding leaves). Older leaves develop an age-related (ontogenic) resistance to powdery mildew and are naturally more resistant to infection than younger leaves. Look for early leaf infections on root suckers, the interior of the canopy or the crotch of the tree where humidity is high. In contrast to other fungi, powdery mildews do not need free water to germinate but germination and fungal growth are favored by high humidity . The disease is more likely to initiate on the undersides (abaxial) of leaves but will occur on both sides at later stages. As the season progresses and infection is spread by wind, leaves may become distorted, curling upward. Severe infections may cause leaves to pucker and twist. Newly developed leaves on new shoots become progressively smaller, are often pale and may be distorted.Early fruit infections are difficult to identify and you will need a hand lens to identify the start of fungal growth in the area where the stem connects to the fruit'],
        'diagnosis':['The FRAC code represents the mode of action of the fungicide. Multiple applications of a fungicide(s) with the same FRAC code increases the risk that resistance will develop. Premix fungicides with two modes of action can improve disease control if there is field resistance to one of the active ingredients and help prevent resistance if there is not. Group 11 fungicides, important for cherry powdery mildew, are high risk for resistance development. It is important for resistance management to use cultural practices to reduce disease pressure. Apply fungicides in a protective rather than reactive manner. Limit the number of applications of a single mode of action both during the season and in sequence (rotate between colors, Figure 7). Apply medium risk compounds no more than three times per season and no more than twice in sequence. High risk compounds should always be alternated with other modes of action. Apply high risk compounds to no more than 30% of the total sprays you use in a single season. Tank mixing with sulfur (or other low risk compounds) can also help limit resistance risk.'],
        'video_url':'https://www.youtube.com/watch?v=N0NJsaCeF8k&pp=ygUfY2hlcnJ5IHBvd2RlcnkgbWlsZGV3IHRyZWF0bWVudA%3D%3D'
    },
    '(including sour) healthy':{
        'description':'This is a healthy green leaf.It does not contain any diseases',
    },
    '(maize) cercospora leaf spot gray leaf spot':{
        'description':'Grey leaf spot (GLS) is a foliar fungal disease that affects maize, also known as corn. GLS is considered one of the most significant yield-limiting diseases of corn worldwide. There are two fungal pathogens that cause GLS: Cercospora zeae-maydis and Cercospora zeina.Symptoms seen on corn include leaf lesions, discoloration (chlorosis), and foliar blight. Distinct symptoms of GLS are rectangular, brown to gray necrotic lesions that run parallel to the leaf, spanning the spaces between the secondary leaf veins. The fungus survives in the debris of topsoil and infects healthy crops via asexual spores called conidia. Environmental conditions that best suit infection and growth include moist, humid, and warm climates.[5][3][4] Poor airflow, low sunlight, overcrowding, improper soil nutrient and irrigation management, and poor soil drainage can all contribute to the propagation of the disease.',
        'symptoms': ['Major outbreaks of grey leaf spot occur whenever favorable weather conditions are present (see § Environment below). The initial symptoms of grey leaf spot emerge as small, dark, moist spots that are encircled by a thin, yellow radiance (lesions formation). The tissue within the “spot" begins to die as spot size increases into longer, narrower leaf lesions. Although initially brownish and yellow, the characteristic grey color that follows is due to the production of grey fungal spores (conidia) on the lesion surface. These symptoms that are similar in shape, size and discoloration are also prevalent on the corn husks and leaf sheaths. Leaf sheath lesions are not surrounded by a yellow radiance, but rather a brown or dark purple radiance.[10] This dark brown or purple discoloration on leaf sheaths is also characteristic to northern corn leaf blight (Exserohilum turcicum), southern corn leaf blight (Bipolaris maydis), or northern corn leaf spot (Bipolaris zeicola). Corn grey leaf spot mature lesions are easily diagnosed and distinguishable from these other diseases. Mature corn grey leaf spot lesions have a brown, rectangular and vein-limited shape. Secondary and tertiary leaf veins limit the width of the lesion and sometimes individual lesions can combine to blight entire leaves.'],
        'diagnosis': ['Resistant varieties-The most proficient and economical method to reduce yield losses from corn grey leaf spot is by introducing resistant plant varieties. In places where leaf spot occurs, these crops can ultimately grow and still be resistant to the disease. Although the disease is not eliminated and resistant varieties show disease symptoms, at the end of the growing season, the disease is not as effective in reducing crop yield. SC 407 have been proven to be common corn variety that are resistant to grey leaf spot.',
                      'Fungicides-Fungicides, if sprayed early in season before initial damage, can be effective in reducing disease.Currently there are 5 known fungicides that treat Corn grey leaf spot If grey leaf spot infection is high, this variety may require fungicide application to achieve full potential. Susceptible varieties should not be planted in previously infected areas'],
        'video_url':'https://www.youtube.com/watch?v=SAp80XsXrJ4&pp=ygU1KG1haXplKSBjZXJjb3Nwb3JhIGxlYWYgc3BvdCBncmF5IGxlYWYgc3BvdCB0cmVhdG1lbnQ%3D'
    },
    '(maize) northern leaf blight':{
        'description':'Northern corn leaf blight (NCLB) or Turcicum leaf blight (TLB) is a foliar disease of corn (maize) caused by Exserohilum turcicum, the anamorph of the ascomycete Setosphaeria turcica. With its characteristic cigar-shaped lesions, this disease can cause significant yield loss in susceptible corn hybrids',
        'symptoms':['The symptoms appear as small, oval, water-soaked spots on the lower leaves first. As the disease progresses, they start to appear on the upper part of the plant. Older spots slowly grow into tan, long cigar-shaped necrotic lesions with distinct dark specks and pale green, water-soaked borders. These lesions later coalesce and engulf a large part of the leaf blade and stalk, sometimes leading to death and lodging. If the infection spreads to the upper parts of the plant during the development of the cob. Severe yield losses can occur (up to 70%).',
                    'The most common diagnostic symptom of the disease on corn is cigar-shaped or elliptical necrotic gray-green lesions on the leaves that range from one to seven inches long. These lesions may first appear as narrow, tan streaks that run parallel to the leaf veins. Fully developed lesions typically have a sooty appearance during humid weather, as a result of spore (conidia) formation. As the disease progresses, the lesions grow together and create large areas of dead leaf tissue. The lesions found in Northern corn leaf blight are more acute if the leaves above the ear are infected during or soon after flowering of the plant.In susceptible corn hybrids, lesions are also found on the husk of ears or leaf sheaths. In partially resistant hybrids, these lesions tend to be smaller due to reduced spore formation. In highly resistant hybrids, the only visible disease symptoms may be minute yellow spots.',
                    'On severely infected plants, lesions can become so numerous that the leaves are eventually destroyed. Late in the season, plants may look like they have been killed by an early frost. Lesions on products containing resistance genes may appear as long, chlorotic, streaks, which can be mistaken for Stewars wilt or Goss wilt'],
        'diagnosis':['E. turcicum causes disease and reduces yield in corn primarily by creating the necrotic lesions and reducing available leaf area for photosynthesis.Following conidia germination, the fungus forms an appressorium, which penetrates the corn leaf cell directly using an infection hypha. Once below the cuticle, the infection hypha produces infection pegs to penetrate the epidermal cell wall. After penetration through the cell wall, the fungus produces intracellular vesicle to obtain nutrients from the cell. After approximately 48 hours after infection, necrotic spots begin to form as the epidermal cells collapse.',
                     'Fungal toxins also play an important role in disease development. Researchers have found that a small peptide called the E.t. toxin allows a non-pathogenic isolate of E. turcicum to infect corn when suspensions of conidia and the toxin were in contact with the leaves. This toxin has also been shown to inhibit root elongation in seedlings and in chlorophyll synthesis. Another toxin produced by E. turcicum, called monocerin, is a lipophilic toxin known to cause necrosis of leaf tissue.',
                     'Organic Control-Bio-fungicides based on Trichoderma harzianum, or Bacillus subtilis can be applied at different stages to decrease the risk of infection. Application of sulfur solutions is also effective',
                     'Chemical Control-An integrated approach with preventive measures together with careful cultural practices is recommended. An early preventative fungicide application can be an effective way of controlling the disease. Otherwise, fungicides can be applied when the symptoms are visible on lower canopy to protect the upper leaves and ears. Apply sprays based on azoxystrobin, picoxystrobin, mancozeb, pyraclostrobin, propiconazole, tetraconazole Apply products based on picoxystrobin + cyproconazole, pyraclostrobin + metconazole, propiconazole + azoxystrobin, prothioconazole + trifloxystrobin Seed treatments are not recommended.'],
        'video_url':'https://www.youtube.com/watch?v=0-iNHLa4Q5c&pp=ygUnKG1haXplKSBub3J0aGVybiBsZWFmIGJsaWdodCcgdHJlYXRtZW50'
    },
    '(maize) healthy':{
        'description':'This is a healthy green leaf.It does not contain any diseases.',
    },
    '(maize) common rust':{
        'description':'Common rust is one of the major foliar diseases in maize, leading to significant grain yield losses and poor grain quality. To dissect the genetic architecture of common rust resistance, a genome-wide association study (GWAS) panel and a bi-parental doubled haploid (DH) population, DH1, were used to perform GWAS and linkage mapping analyses. The GWAS results revealed six single-nucleotide polymorphisms (SNPs) significantly associated with quantitative resistance of common rust at a very stringent threshold of P-value 3.7006 ',
        'symptoms':['The typical developed rust symptoms on leaves show circular to elongate (0.2 to 2 mm long) with dark brown pustule (uredinia) scattered over both leaf surfaces giving the leaf a rusty appearance. Pustules emerge in circular bands due to infection that occurred in the whorl. Pustules break through the leaf epidermis and release powdery reddish-brown spores (urediospores). As pustules mature, they release brownish-black spores (teliospores) which are the overwintering spores. Under severe disease pressure, leaves turn chlorotic and dry prematurely (Plate 1)'],
        'diagnosis':['t inhibition of developmentof Puccinia striiformis in wheat leaf tissues treated with the fungicide was accompanied by severe morphologicaland structural changes in the hyphal and haustorialdevelopment. These changes included increased vacuolation, irregular cell wall thickening and necrosis ordegeneration of cytoplasm. These alterations are versimilar to those reported for other plant pathogenic fungitreated with ergosterol biosynthesis-inhibiting (EBI) fungicides (Coutinho et al., 1995; Leinhos et al., 1997). Morphological alterations of hyphal structures andhaustoria of the stripe rust fungus in tebuconazole treated wheat plants may be triggered by the primary mode of action of triazole fungicides. Interference in sterol biosynthesis by inhibition of 14a-demethylase results in insufficient availability of ergosterol and accumulation of 14a-methyl sterols. Ergosterol, an essential membrane constituent, may be responsible for maintaining membraneintegrity and activity. Insufficiency of ergosterol in fungalmembranes severely disturbs membrane functions'],
        'vieo_url':'https://www.youtube.com/watch?v=KUXGyLtu84Q&pp=ygUbbWFpemUgY29tbW9uIHJ1c3QgdHJlYXRtZW50'
    },
    'grape black rot':{
        'description':'Grape black rot is a fungal disease caused by an ascomycetous fungus, Guignardia bidwellii, that attacks grape vines during hot and humid weather. “Grape black rot originated in eastern North America, but now occurs in portions of Europe, South America, and Asia. It can cause complete crop loss in warm, humid climates, but is virtually unknown in regions with arid summers.”The name comes from the black fringe that borders growing brown patches on the leaves. The disease also attacks other parts of the plant, “all green parts of the vine: the shoots, leaf and fruit stems, tendrils, and fruit. The most damaging effect is to the fruit”.Grape black rot affects many grape growers throughout the United States, therefore, it is important to understand the disease life cycle and environmental conditions to best manage the disease. Once infection takes place, different methods are available to control the disease.',
        'symptoms':['Infection spots (lesions) usually do not appear on fruits until after they are half grown. The first symptom is the appearance of very small whitish areas on developing green fruit. These are soon surrounded by a rapidly widening brown ring, giving a birds eye effect. As the fruit rots, black sexual fruiting bodies, perithecia, begin to form. Later, the rotted fruits become black, perithecia studded mummies. Mummies are easily dislodged but some may remain attached through the winter. After spring rains thoroughly soak the mummies, spore-bearing bodies (asci) in the perithecia forcibly release ascospores. Air movements carry ascospores to developing plant parts, which they infect under wet conditions.'],
        'diagnosis':['Sanitation and Cultural Practices-Pruning of vines removes much infected tissue, but the pruning action causes old fruit mummies to fall to the ground, thus providing them an advantageous environment to become wetted for long periods, thus aiding spore production. Cultivation which throws soil over the mummies prevents release of spores. Weed control and proper pruning provide good air circulation, thus keeping foliage dry and protected from fungus infection.',
                     'Resistant Cultivars (Varieties)-Many grape cultivars have some resistance but still require a few applications of recommended fungicide sprays during wet periods. Cultivars that have large fruits or mature early in the season are the most susceptible.',
                     'Chemical Control-Certain fungicides are effective for preventing black rot if they are applied beginning early in the spring when the young shoots are developing and spraying is timed to anticipate a rainy period.'],
        'video_url':'https://www.youtube.com/watch?v=aOtNRrHHwds&pp=ygUZZ3JhcGUgYmxhY2sgcm90IHRyZWF0bWVudA%3D%3D'
    },
    'esca(black measles)':{
        'description':'Grapevine measles, also called esca, black measles or Spanish measles, has long plagued grape growers with its cryptic expression of symptoms and, for a long time, a lack of identifiable causal organism(s). The name measles refers to the superficial spots found on the fruit. ',
        'symptoms':['The disease may occur at any time during the growing season. The main symptom is an interveinal "striping" on the leaves, which is characterized by the discoloration and drying of the tissues around the main veins. It usually shows as dark red in red varieties and yellow in white ones. Leaves can dry out completely and drop prematurely. On berries, small, round, dark spots, often bordered by a brown-purple ring, may occur. These fruit spots may appear at any time between fruit set and ripening. In severely affected vines, the berries often crack and dry. Cross-sectional cuts through affected canes, spurs, cordons, or trunks reveal concentric rings formed by dark spots. A severe form of Esca known as apoplexy" results in a sudden dieback of the entire vine.'],
        'diagnosis':['Organic Control-Soak dormant cuttings for 30 mins in hot water at about 50°C. This treatment is not always effective and must therefore be combined with other methods. Some species of Trichoderma have been used to prevent the infection of pruning wounds, basal ends of propagation material, and graft unions. This treatment has to be carried out within 24 hours of pruning and again 2 weeks after.',
                     'Chemical Control-Always consider an integrated approach with preventive measures together with biological treatments if available. Chemical strategies to control this disease are difficult since the traditional wound protectants do not penetrate deep enough in the dormant grapevine cuttings to affect the fungi. Preventive practices are the most effective management approach for all trunk diseases. For example, immediately before grafting vines can be deeper into specialized waxes containing plant growth regulators or fungicide impregnated formulations. This encourages graft union callus development while inhibiting fungal contamination.'],
        'video_url':'https://www.youtube.com/watch?v=aOtNRrHHwds&pp=ygUtZ3JhcGUgYmxhY2sgbWVhc2xlcyB0cmVhdG1lbnQgYW5kIGZlcnRpbGl6ZXJz'
    },
    'leaf blight(isariopsis leaf spot)':{
        'description':'Diseased leaves appear whitish gray, dusty, or have a powdery white appearance. Petioles, cluster stems, and green shoots often look distorted or stunted. Berries can be infected until their sugar content reaches about 8%. If infected when young, the epidermis of the berry.',
        'symptoms':['The disease attacks both leaves and fruits. Small yellowish spots first appear along the leaf margins, which gradually enlarge and turn into brownish patches with concentric rings. Severe infection leads to drying and defoliation of leaves. Symptoms in the form of dark brown-purplish patches appear on the infected berries, rachis and bunch stalk just below its attachment with the shoots.'],
        'diagnosis':['Survival and spread-The disease is externally and internally seed borne. The pathogen survives through spores (conidia) or mycelium in diseased plant debris or weed.',
                     'Favourable conditions-Moist (More than 70% relative humidity) and warm weather (12-25 ºC) and intermittent rains favours disease development.'],
        'video_url':'https://www.youtube.com/watch?v=agIwEBm7Zao&pp=ygUbR1JBUEUgTEVBRiBCTElHSFQgdHJlYXRtZW50'
    },
    'healthy':{
        'description':'This is a healthy green leaf.It does not contain any diseases.'
    },
    'haunglongbing(citrus greening)':{
        'description':'Orange Huanglongbing (HLB), also known as citrus greening disease, is a bacterial disease that affects citrus trees. The disease is caused by the bacterium Candidatus Liberibacter asiaticus, which is transmitted by a tiny insect called the Asian citrus psyllid.',
        'symptoms':['1. Yellow shoots and leaves',
                    '2. Uneven ripening of fruit',
                    '3. Misshapen and small fruit',
                    '4. Premature fruit drop',
                    '5. Lopsided fruit'],
        'diagnosis':['1. Regularly monitor trees for signs of HLB and remove any infected trees immediately.',
'2. Control the Asian citrus psyllid population through insecticide treatments and proper disposal of infected plant material.',
'3. Use certified disease-free citrus nursery stock for new plantings.',
'4. Implement a nutrient management program to maintain tree health and vigor.',
'5. Promote proper irrigation and drainage to prevent water stress, which can make trees more susceptible to HLB.',
'6. Quarantine and restrict the movement of citrus plants, fruit, and plant material from infected areas.',
'7. Practice good hygiene and sanitation when working with citrus trees to avoid spreading the disease.'],
        'video_url':'https://www.youtube.com/watch?v=wFjjSOgEqVQ&pp=ygUeT1JBTkdFIEhBVU5HTE9OR0JJTkcgdHJlYXRtZW50'
         },
    'bacterial spot':{
        'description':'Peach bacterial spot disease is caused by the bacterium Xanthomonas arboricola pv. pruni and can cause significant damage to peach trees, reducing fruit yield and quality. Here are some symptoms and prevention methods for this disease',
        'symptoms':['Small, dark spots with yellow halos appear on leaves, twigs, and fruit.',
                   'Spots on leaves may coalesce, causing defoliation.',
                   'Fruit spots may lead to cracking, reducing quality and making them more susceptible to rot.',
                   'In severe cases, cankers may form on branches and trunks, leading to dieback.'],
        'diagnosis':['Plant disease-resistant cultivars when possible.',
'Use clean planting materials, and avoid planting in areas where the disease has previously occurred.',
'Use drip irrigation rather than overhead sprinklers, which can spread the bacteria.',
'Prune trees to promote good air circulation and sunlight penetration, which can reduce disease incidence.',
'Remove and destroy any infected plant material, including fallen fruit and leaves.',
'Apply copper-based fungicides during the dormant season to help prevent infection. However, copper should be used with caution as it can harm beneficial insects and soil microbes.'],
        'video_url':'https://www.youtube.com/watch?v=LuB6RZRWCtQ&pp=ygUeUEVBQ0ggQkFDVEVSSUFMIFNQT1QgdHJlYXRtZW50'
        },
    'healthy':{
        'description':'This is a healthy green leaf.It does not contain any diseases.'
    },
    'bacterial spot':{
        'description': 'Pepper plants can be affected by bacterial spot disease, which is caused by the bacterium Xanthomonas campestris. The disease can cause significant damage to pepper crops if left untreated. ',
        'symptoms':['1. Dark spots on the leaves, which can be circular or irregular in shape.',
'2. Lesions on the fruit, which can be sunken or raised and may be accompanied by a yellow halo.',
'3. Leaf drop and defoliation.',
'4. Stunted growth and reduced yield.'],
        'diagnosis':['1. Plant resistant varieties: Choose pepper varieties that are resistant to bacterial spot disease.',
'2. Maintain proper sanitation: Keep the garden clean and free from plant debris and weeds.',
'3. Use disease-free seeds and plants: Buy seeds and plants from reputable sources and inspect them before planting.',
'4. Water properly: Avoid overhead watering and use drip irrigation or a soaker hose to keep the foliage dry.',
'5. Apply copper-based fungicides: Copper-based fungicides can help control bacterial spot disease on pepper plants. Follow the manufacturers instructions when applying fungicides.',
'6. Rotate crops: Avoid planting pepper plants in the same spot year after year, as this can increase the risk of disease.',
'7. Practice crop management: Remove and destroy infected plants and fruits as soon as they are detected to prevent the spread of the disease.'],
        'video_url':'https://www.youtube.com/watch?v=1HgsMF4gd7U&pp=ygUkUEVQUEVSIEJFTEwgQkFDVEVSSUFMIFNQT1QgdHJlYXRtZW50'
    },
    'healthy':{
        'description':'This is a healthy green leaf.It does not contain any diseases.'
    },
    'early blight':{
        'description':'Potato early blight is a common fungal disease that affects potato plants, caused by the fungus Alternaria solani. Here are some symptoms and prevention measures.',
        'symptoms':['Brown to black circular spots with yellow halos on leaves, stems, and tubers.',
'Leaves turn yellow and drop prematurely, reducing the plants ability to produce energy.',
'If the disease progresses, the spots on the leaves merge, causing the leaves to wither and die.'],
        'diagnosis':['Crop rotation: Do not plant potatoes or any other solanaceous plants in the same location for two to three years.',
'Proper sanitation: Remove and destroy any infected plant debris, as the fungus can survive in plant debris for long periods and infect subsequent crops.',
'Fungicides: Apply fungicides at the first sign of the disease. Fungicides containing chlorothalonil or copper-based compounds are commonly used for early blight.',
'Plant resistant varieties: Choose potato varieties that are resistant to early blight.',
'Proper watering: Avoid overhead watering, as wet foliage can promote the spread of the fungus. Instead, use drip irrigation or water at the base of the plant.'],
        'video_url':'https://www.youtube.com/watch?v=6i5_sLY_pWc&pp=ygUdUE9UQVRPIEVBUkxZIEJMSUdIVCB0cmVhdG1lbnQ%3D'
        },
    'late blight':{
        'description':'Potato late blight is a serious fungal disease caused by Phytophthora infestans that affects potato plants. The disease can spread rapidly and cause significant damage to crops. ',
        'symptoms':['1. Brown to black lesions on the leaves, stems, and tubers',
'2. Dark, water-soaked spots on the leaves',
'3. White fungal growth on the undersides of the leaves, especially in wet weather',
'4. Rapid defoliation of the plant',
'5. Rotting of the tubers, which appear shrunken and discolored'],
        'diagnosis':['1. Choose healthy seed potatoes that are certified disease-free.',
'2. Plant potatoes in well-drained soil, and avoid planting in areas that have previously been affected by late blight.',
'3. Monitor your potato plants regularly for signs of disease, and remove and destroy any infected plants immediately.',
'4. Space plants properly to ensure good air circulation, which will help to prevent the spread of disease.',
'5. Apply fungicides preventively, according to label instructions, especially during periods of high humidity and rainfall.'],
        'video_url':'https://www.youtube.com/watch?v=mHrcs1c_ToA&pp=ygUbUE9UQVRPIExBVEUgQkxJR0hUdHJlYXRtZW50'
    },
    'healthy':{
        'description':'This is a healthy green leaf.It does not contain any diseases.',
    },
    'healthy':{
        'description':'This is a healthy green leaf.It does not contain any diseases.'
    },
    'healthy':{
        'description':'This is a healthy green leaf.It does not contain any diseases.'
    },
    'powdery mildew':{
        'description':'Powdery mildew is a fungus that attacks many types of vegetable crops including the squash plant. Of the eight species of powdery mildew, Erysiphe cichoracearum is the species that attack squash plants.',
        'symptoms':['Leaves may gradually turn completely yellow, die, and fall off.',
'Leaves may twist.',
'Leaves may buckle.',
'May shorten production time.',
'May reduce fruit yields.',
'May produce fruit with less flavor.'],
        'diagnosis':['Organic Milk Fungicide-Organic milk that has not been pasteurized can control the severity of powdery mildew. To make this organic fungicide, mix one part organic milk with nine parts water. Pour mixture into a garden sprayer. Once a week spray the entire plant including the underside of leaves.',
                    'Homemade Baking Soda Spray-Baking soda combined with a dormant oil and liquid soap is effective against powdery mildew if it is applied prior to or early in an outbreak of fungus. Combine these ingredients to make your mixture: one tablespoon of baking soda, one teaspoon of dormant oil, one teaspoon of insecticidal or liquid soap (not detergent), and one gallon of water. Apply weekly.',
                    'Potassium Bicarbonate-Scientific studies have proven that potassium bicarbonate is an effective substance in killing powdery mildew. Here are the ingredients to make this effective fungicide: three tablespoons of potassium bicarbonate, three tablespoons of vegetable oil, a half teaspoon of soap, one gallon of water.'],
        'video_url':'https://www.youtube.com/watch?v=_IRM4iGHGic&pp=ygUfc3F1YXNoIHBvd2RlcnkgbWlsZGV3IHRyZWF0bWVudA%3D%3D'
    },
    'leaf scorch':{
        'description':'Diplocarpon earlianum is a species of fungus that causes disease in strawberry plants called strawberry leaf scorch. The disease overwinters in plant debris and infects strawberry plants during the spring season when it is wet. The five main methods to reduce strawberry leaf scorch include: irrigation techniques, crop rotation, planting resistant and disease-free seeds, fungicide use, and sanitation measures. Control of strawberry leaf scorch is important because it is responsible for the majority of disease in strawberries. Diplocarpon earliana affects the fruit quality and yield of the strawberry crop',
        'symptoms':['The disease is characterized by numerous small, purplish to brownish lesions (from 1/16 to 3/16 of an inch in diameter) with undefined borders on the upper surface of the leaf. These symptoms are different from strawberry leaf spot which has brown lesions with defined borders and a lighter center. As the leaf scorch progresses over time, the leaves turn brown and dry up, resembling a burnt or “scorched” appearance as indicated by its disease name. It is common for the petioles of the leaves to have purple, sunken lesions that resemble streaks. If these streaks are severe enough, they may lead to the bowing of the petiole which in turn kills the leaf. Strawberry leaf scorch infects all parts of the flower, leading to unattractive blemishes on the fruit (strawberries).',
'Minuscule dark, black spots are a sign of the fungus. These spots are specialized asexual fruiting bodies called acervuli. When the acervuli accumulate into masses, they resemble little drops of tar. Very rarely, the sexual structure apothecia that develop in advanced lesions of the plant can be seen.[4] Leaf scorch seriously weakens the host plant, greatly reducing the ability to tolerate drought stress and the ability and lowering resistance to winter damage'],
        'diagnosis':['Planting resistant and disease-free seed as well as burning all plant debris after harvest are common sanitation methods used. Resistant varieties of strawberry plants will be able to grow and produce fruit with limited effects of D. earliana. Disease-free seed allows the new and emerging strawberry plants an increased chance of producing undiseased fruit. Lastly, burning the plant debris left after harvest decreases the amount of the D. earliana inoculum present in the subsequent season of production.Crop rotation can be used in intervals of three to five years. Crop rotation gives various nutrients a chance to accumulate in the soil, such as nitrogen, as well as the mitigation of pests or in the case of D. earliana, pathogens. For this reason, the crop rotated with strawberries should not be a host for D. earliana. Since the rotated crop is not a host for D. earliana, the pathogen has a severely decreased chance of survival in structures such as endospores. Some common crops in this rotation include corn and legumes, which can increase soil quality, suppress the strawberry leaf scorch pathogen, and reduce the amount of weeds. Frequent renewal of strawberry plantings helps to prevent severe scorch because the disease often does not become severe during the first and second years after planting Annual cropping systems are observed to have much lower risk and occurrences of infection. Symptoms may be present but will generally disappear before the disease can progress. Generally, an annual strawberry system will not need further disease management action and economic losses are not of heavy concern.Fungicides, such as thiophanate-methyl, are used to inhibit the ability of D. earliana to access the host. Therefore, it prevents the growth of the fungus on the strawberry leaves. These fungicides are applied a variety of ways, at intervals ranging from one to two weeks when the strawberry plants are in early bloom. The number of applications depends on the extent of the disease the previous year as well as the water conditions during application. An increase in the wetness of the environment would lead to an increase in the number of applications of the fungicide'],
        'video_url':'https://www.youtube.com/watch?v=2H8mqgOw9PY&pp=ygUiU3RyYXdiZXJyeV9fX0xlYWZfc2NvcmNoIHRyZWF0bWVudA%3D%3D'
    },
    'healthy':{
        'description':'This is a healthy green leaf.It does not contain any diseases.'
    },
    'bacterial spot':{
        'description':'Bacterial spot of tomato is a potentially devastating disease that, in severe cases, can lead to unmarketable fruit and even plant death.  Bacterial spot can occur wherever tomatoes are grown, but is found most frequently in warm, wet climates, as well as in greenhouses.  The disease is often an issue in Wisconsin.',
        'symptoms':['Bacterial spot can affect all above ground parts of a tomato plant, including the leaves, stems, and fruit.  Bacterial spot appears on leaves as small (less than ⅛ inch), sometimes water-soaked (i.e., wet-looking) circular areas.  Spots may initially be yellow-green, but darken to brownish-red as they age.  When the disease is severe, extensive leaf yellowing and leaf loss can also occur.  On green fruit, spots are typically small, raised and blister-like, and may have a yellowish halo.  As fruit mature, the spots enlarge (reaching a maximum size of ¼ inch) and turn brown, scabby and rough.  Mature spots may be raised, or sunken with raised edges.  Bacterial spot symptoms can be easily confused with symptoms of another tomato disease called bacterial speck.  For more information on this disease, see University of Wisconsin Garden Facts XHT1250.'],
        'diagnosis':['Plant pathogen-free seed or transplants to prevent the introduction of bacterial spot pathogens on contaminated seed or seedlings.  If a clean seed source is not available or you suspect that your seed is contaminated, soak seeds in water at 122°F for 25 min. to kill the pathogens.  To keep leaves dry and to prevent the spread of the pathogens, avoid overhead watering (e.g., with a wand or sprinkler) of established plants and instead use a drip-tape or soaker-hose.  Also to prevent spread, DO NOT handle plants when they are wet (e.g., from dew) and routinely sterilize tools with either 10% bleach solution or (better) 70% alcohol (e.g., rubbing alcohol).  Where bacterial spot has been a recurring problem, consider using preventative applications of copper-based products registered for use on tomato, especially during warm, wet periods.  Keep in mind however, that if used excessively or for prolonged periods, copper may no longer control the disease.  Be sure to read and follow all label instructions of the product that you select to ensure that you use it in the safest and most effective manner possible.  Burn, bury or hot compost tomato debris at the end of the season.  Wait at least one year before planting tomatoes in a given location again, and remove and burn, bury or hot compost any volunteer tomatoes that come up in your garden.'],
        'video_url':'https://www.youtube.com/watch?v=jONPVKSvFW0&pp=ygUfdG9tYXRvIGJhY3RlcmlhbCBzcG90IHRyZWF0bWVudA%3D%3D'
    },
    'early blight':{
        'description':'Tomato blight is a disease caused by a fungus, depending on which type of blight is affecting the vegetable. However, more than one type of blight can attack tomatoes at the same time. There are three types of tomato blight caused by different fungi that all present somewhat similarly',
        'symptoms':['Early blight is sometimes confused with Septoria leaf spot. They both form spots on the leaves, which eventually turn yellow and die off, but Septoria also forms fruiting bodies that look like tiny filaments coming from the spots.',
'On Older Plants: Dark spots with concentric rings develop on older leaves first. The surrounding leaf area may turn yellow. Affected leaves may die prematurely, exposing the fruits to sun-scald.',
'Dark lesions on the stems start off small and are slightly sunken. As they get larger, they elongate and you will start to see concentric markings like the spots on the leaves. Spots that form near ground level can cause some girdling of the stem or collar rot. Plants may survive, but they will not thrive or produce many tomatoes.',
'On Tomato Fruits: If early blight gets on the fruits, spots will begin at the stem end, forming a dark, leathery, sunken area with concentric rings. Both green and ripe tomatoes can be affected.',
'On Seedlings: Affected seedlings will have dark spots on their leaves and stems. They may even develop the disease on their cotyledon leaves. Stems often wind up girdled.'],
        'diagnosis':['Certified Seed: Buy seeds and seedlings from reputable sources and inspect all plants before putting them in your garden.',
'Garden Sanitation: Since early blight can overwinter on plant debris and in the soil, sanitation is essential. So many tomato diseases can come into your garden this way so its very important to clean up all plant residue at the end of the season.',
'Rotate Crops: If you have an outbreak of early blight, find somewhere else to plant your tomatoes next year, even if its in containers.',
'Separate Plants of the Same Family: Do not grow nightshade (Solanum) plants, such as eggplant, potatoes, and peppers alongside tomatoes to avoid passing along the same infections to each other. Instead, be aware of the correct companion plantings with tomatoes to avoid blight.'],
        'video_url':'https://www.youtube.com/watch?v=2GYD7aVBFtg&pp=ygUedG9tYXRvIGVhcmx5IGJsaWdodHQgdHJlYXRtZW50'
    },
    'late blight':{
        'description':'Tomato late blight is caused by the oomycete pathogen Phytophthora infestans (P. infestans). The pathogen is best known for causing the devastating Irish potato famine of the 1840s, which killed over a million people, and caused another million to leave the country.',
        'symptoms':['The first symptoms of late blight on tomato leaves are irregularly shaped, water-soaked lesions, often with a lighter halo or ring around them ; these lesions are typically found on the younger, more succulent leaves in the top portion of the plant canopy. During high humidity, white cottony growth may be visible on underside of the leaf , where sporangia form . Spots are visible on both sides of the leaves. As the disease progresses, lesions enlarge causing leaves to brown, shrivel and die. Late blight can also attack tomato fruit in all stages of development. Rotted fruit are typically firm with greasy spots that eventually become leathery and chocolate brown in color ; these spots can enlarge to the point of encompassing the entire fruit.'],
        'diagnosis':['Disease Control for Organic Growers-Skip to Disease Control for Organic Growers',
'Organic growers have fewer chemical options that are effective; the only OMRI labeled active ingredients that have decent efficacy against late blight are fixed copper formulations. Organic growers should plant susceptible varieties early in the season or select a late blight resistant variety.',
'Disease Control for Home Gardeners-Skip to Disease Control for Home Gardeners Products containing the active ingredients copper or chlorothalonil (the trade name of one product with chlorothalonil is known as ‘Daconil’) are the best and only effective products available to home gardeners. In addition, home gardeners should grow varieties with resistance if they are worried about late blight in future years because most chemicals available to the home gardener are not sufficient to control late blight once it appears. Once plants are infected in a home garden, there is little that can be done to protect them besides weekly fungicide sprays.',
'Home owners should plant susceptible varieties early in the season or select a late blight resistant variety.'],
        'video_url':'https://www.youtube.com/watch?v=kY_Q8d8GMN4&pp=ygUddG9tYXRvIGxhdGUgYmxpZ2h0dCB0cmVhdG1lbnQ%3D'
    },
    'leaf mold':{
        'description':'Cladosporium fulvum is an Ascomycete called Passalora fulva, a non-obligate pathogen that causes the disease on tomato known as the tomato leaf mold. P. fulva only attacks tomato plants, especially the foliage, and it is a common disease in greenhouses, but can also occur in the field.',
        'symptoms':['Symptoms of tomato leaf mold appear usually with foliage, but fruit infection is rare. The primary symptom appear on the upper surface of infected leaves as a small spot pale green or yellowish with indefinite margins, and on corresponding area of the lower surface, the fungus begins to sporulate.[5] The diagnostic symptom develops on lower surface as an olive green to grayish purple and velvety appearance, which are composed of spores (conidia).[6] Continuously, the color of the infected leaf changes to yellowish brown and the leaf begins to curl and dry. The leaves will drop upon reaching a premature stage, and the defoliation of the infected host will cause further infection. This disease develops well in relative humidity levels above 85%. When the temperature reaches optimum level for germinating, the host will be infected by the pathogen. Occasionally, this pathogen causes disease on the fruit or blossoms with various symptoms.[7] Fruits such as green and ripe one will develop dark rot on the stem. The blossoms will be killed before fruits grow.'],
        'diagnosis':['The disease management or control can be divided into two main groups: disease control in greenhouse and disease control in the field. Both controls are very similar. The differences are presented in few controls adopted in greenhouse in which some environmental conditions are controlled such as humidity and temperature as well sanitization of the greenhouse.',
'Culture-The first strategy of management is the cultural practices for reducing the disease. It includes adequating row and plant spacing that promote better air circulation through the canopy reducing the humidity; preventing excessive nitrogen on fertilization since nitrogen out of balance enhances foliage disease development; keeping the relatively humidity below 85% (suitable on greenhouse), promote air circulation inside the greenhouse, early planting might to reduce the disease severity and seed treatment with hot water (25 minutes at 122 °F or 50 °C).',
'Sanitation-The second strategy of management is the sanitization control in order to reduce the primary inoculum. Remove and destroy (burn) all plants debris after the harvest, scout for disease and rogue infected plants as soon as detected and steam sanitization the greenhouse between crops.',
'Resistance-The most effective and widespread method of disease control is to use resistant cultivars. However, only few resistant cultivar to tomato leaf mold are known such as Caruso, Capello, Cobra (race 5), Jumbo and Dombito (races 1 and 2). Moreover, this disease is not considered an important disease for breeding field tomatoes.',
'Chemical control-The least but not the less important management is the chemical control that ensure good control of the disease. The chemical control is basically spraying fungicide as soon as the symptoms are evident. Compounds registered for using are: chlorothalonil, maneb, mancozeb and copper.'],
        'video_url':'https://www.youtube.com/watch?v=0lZOboTH8m4&pp=ygUadG9tYXRvIGxlYWYgbW9sZCB0cmVhdG1lbnQ%3D'
    },
    'septoria leaf spot':{
        'description':'eptoria leaf spot is one of two common fungal diseases that can devastate tomatoes in both commercial settings and home gardens.  The second common tomato blight, early blight, is detailed in University of Wisconsin Garden Facts XHT1074.',
        'symptoms':[' Symptoms of Septoria leaf spot first appear at the base of affected plants, where small (approximately ¼ inch diameter) spots appear on leaves and stems.  These spots typically have a whitish center and a dark border.  Eventually multiple spots on a single leaf will merge, leading to extensive destruction of leaf tissue.  Septoria leaf spot can lead to total defoliation of lower leaves and even the death of an infected plant.'],
        'diagnosis':['Septoria leaf spot is best controlled using preventative measures.  Destroy infested plants by burning or burying them.  Rotate vegetables to different parts of your garden each year to avoid areas where infested debris (and thus spores of Septoria lycopersici) may be present.  Use Septoria leaf spot-resistant tomato varieties whenever possible.  Increase spacing between plants to increase airflow and decrease humidity and foliage drying time.  Mulch your garden with approximately one inch of a high quality mulch, but DO NOT overmulch as this can lead to wet soils that can contribute to increased humidity.  Finally, where the disease has been a chronic problem, use of preventative applications of a copper or chlorothalonil-containing fungicide labeled for use on vegetables may be warranted.'],
        'video_url':'https://www.youtube.com/watch?v=bI0B4IsQT3w&pp=ygUlVG9tYXRvX19fU2VwdG9yaWFfbGVhZl9zcG90IHRyZWF0bWVudA%3D%3D'
    
    },
    'spider mites two spotted spider mite':{
        'description':'The two-spotted spider mite is a common species on tomatoes in the south and is distinguishable by a pair of dark spots visible through the orange body. The dark spots on spider mites are generally the accumulation of body waste under the skin, hence the newly formed individuals may lack the spots.',
        'symptoms':['As the summer heat continues, it is common to see spider mites on commercial and home-grown tomatoes. Heat, drought, water stress, the presence of a large number of weeds, and incorrect use of insecticides can lead to high buildup of mites on tomatoes.Mites are 1/50 inch in diameter and usually feed on the underside of leaves close to the midrib. There are several species of spider mites and they typically have a short life cycle of seven to eight days. Eggs are attached to the webbing produced by the adults, and the nymphal stages have three pairs of legs (i.e., resembling an insect).'],
        'diagnosis':[' Abamectin (AgriMek, Syngenta Crop Protection, 8 to 16 fluid ounces per acre): Abamectin is actually a naturally derived substance and serves as a good rescue insecticide. AgriMek contains synthetic abamectin and provides long-term residual control of two-spotted spider mites.',
'AgriMek also has a locally systemic action or translaminar activity and must be tank-mixed with a surfactant to enable the product to move inside the leaves. Do not apply more than two sequential applications of abamectin to tomatoes to prevent resistance buildup. The postharvest interval on tomatoes is seven days.',
'Bifenazate (Acramite, Chemtura, 0.75 to 1 pound per acre): This is a contact poison against two-spotted spider mites with less toxicity to predaceous mites and beneficial arthropods. Acramite is a good knockdown product and also kills the eggs (ovicidal action). Do not make more than one application of bifenazate per season. The postharvest interval on tomatoes is three days.',
'Spiromesifen (Oberon, Bayer CropScience, 7 to 8.5 ounces per acre) and fenpyroximate (Portal, Nichino America, 2 pints per acre): These are slow acting contact poisons. Oberon, a lipid biosynthesis inhibitor in insects, may also provide some whitefly control. These products can take up to four days to activate after application and two applications may be necessary to get control of a spider mite outbreak.'],
        'video_url':'https://www.youtube.com/watch?v=bqCBIP9TmcY&pp=ygUrdG9tYXRvIHNwaWRlciBtaXRlcyB0d28gc3BvdHRlZCBzcGlkZXIgbWl0ZQ%3D%3D'
    },
    'target spot':{
        'description':'Corynespora cassiicola is a species of fungus well known as a plant pathogen. It is a sac fungus in the family Corynesporascaceae. It is the type species of the genus Corynespora',
        'symptoms':['Target spot on tomato fruit is difficult to recognize in the early stages, as the disease resembles several other fungal diseases of tomatoes. However, as diseased tomatoes ripen and turn from green to red, the fruit displays circular spots with concentric, target-like rings and a velvety black, fungal lesion in the center. The “targets” become pitted and larger as the tomato matures.Read more at Gardening Know How: Target Spot On Tomato Fruit Tips On Treating Target Spot On Tomatoes'],
        'diagnosis':['If the Corynespora cassiicola is discovered on the plant during its development, management of the disease includes removing and burning the plants lower leaves. Additionally, its important to ensure that there are no weeds present on the plant plots because these weeds may act as hosts and harbor the fungus. Additionally, weeds can be considered disadvantageous in a field because they work to compete against the host for nutrients. Some tactics for managing weeds include applying mulch to the soil or introducing a natural pathogen of the weed as a method of biocontrol. If the pathogen is discovered after harvesting the host, management includes burning the infected crop in the attempt to rid the disease from the environment. Furthermore, practicing plant rotation and waiting three years before replanting the host on the same land can be beneficial for pathogen prevention'],
        'video_url':'https://www.youtube.com/watch?v=b4JRKafAu-Q&pp=ygUddG9tYXRvIHRhcmdldCBzcGl0ZSB0cmVhdG1lbnQ%3D'
    },
    'tomato yellow leaf curl virus':{
        'description':'Tomato yellow leaf curl virus (TYLCV) is a DNA virus from the genus Begomovirus and the family Geminiviridae. TYLCV causes the most destructive disease of tomato, and it can be found in tropical and subtropical regions causing severe economic losses. This virus is transmitted by an insect vector from the family Aleyrodidae and order Hemiptera, the whitefly Bemisia tabaci, commonly known as the silverleaf whitefly or the sweet potato whitefly. The primary host for TYLCV is the tomato plant, and other plant hosts where TYLCV infection has been found include eggplants, potatoes, tobacco, beans, and peppers.[1] Due to the rapid spread of TYLCV in the last few decades, there is an increased focus in research trying to understand and control this damaging pathogen. Some interesting findings include virus being sexually transmitted from infected males to non-infected females (and vice versa), and an evidence that TYLCV is transovarially transmitted to offspring for two generations',
        'symptoms':['The most obvious symptoms in tomato plants are small leaves that become yellow between the veins. The leaves also curl upwards and towards the middle of the leaf',
                    'In seedlings, the shoots become shortened and give the young plants a bushy appearance. In mature plants only new growths produced after infection is reduced in size. Although tomato production is reduced by the infection, the fruit appears unaffected.'],
        'diagnosis':['Currently, the most effective treatments used to control the spread of TYLCV are insecticides and resistant crop varieties. The effectiveness of insecticides is not optimal in tropical areas due to whitefly resistance against the insecticides; therefore, insecticides should be alternated or mixed to provide the most effective treatment against virus transmission.[6] Developing countries experience the most significant losses due to TYLCV infections due to the warm climate as well as the expensive costs of insecticides used as the control strategy. Other methods to control the spread of TYLCV include planting resistant/tolerant lines, crop rotation, and breeding for resistance of TYLCV. As with many other plant viruses, one of the most promising methods to control TYLCV is the production of transgenic tomato plants resistant to TYLCV'],
        'video_url':'https://www.youtube.com/watch?v=-KTdzdFuPL8&pp=ygUodG9tYXRvIHllbGxvdyBsZWFmIGN1cmwgdmlydXMgdHJlYXRtZW50IA%3D%3D'
    },
    'tomato mosaic virus':{
        'description':'Besides Solanaceous plants, such as pepper and petunia, ToMV affects a wide range of other crop and ornamental plants. These include snapdragon, delphinium and marigold and a great many other plants to a lesser extent. The infection is generally restricted to plants that are grown in seedbeds and transplanted as it is in the handling processes that the virus is likely to gain entry. Symptoms on other plant hosts include blistering, chlorosis, curling, distortion, dwarfing and mottling of the leave',
        'symptoms':['Besides Solanaceous plants, such as pepper and petunia, ToMV affects a wide range of other crop and ornamental plants. These include snapdragon, delphinium and marigold and a great many other plants to a lesser extent. The infection is generally restricted to plants that are grown in seedbeds and transplanted as it is in the handling processes that the virus is likely to gain entry. Symptoms on other plant hosts include blistering, chlorosis, curling, distortion, dwarfing and mottling of the leave'],
        'diagnosis':['Fungicides will NOT treat this viral disease.',
'Plant resistant varieties when available or purchase transplants from a reputable source.',
'Do NOT save seeds from infected crops.',
'Spot treat with least-toxic, natural pest control products, such as Safer Soap, Bon-Neem, and diatomaceous earth, to reduce the number of disease-carrying insects.',
'Harvest-Guard® row cover will help keep insect pests off vulnerable crops/ transplants and should be installed until bloom.',
'Remove all perennial weeds, using least-toxic herbicides, within 100 yards of your garden plot.',
'The virus can be spread through human activity, tools, and equipment. Frequently wash your hands and disinfect garden tools, stakes, ties, pots, greenhouse benches, etc. (one part bleach to 4 parts water) to reduce the risk of contamination.',
'Avoid working in the garden during damp conditions (viruses are easily spread when plants are wet).',
'Avoid using tobacco around susceptible plants. Cigarettes and other tobacco products may be infected and can spread the virus.',
'Remove and destroy all infected plants (see Fall Garden Cleanup). Do NOT compost'],
        'video_url':'https://youtu.be/QdBkR12jxKU'
    },
    'healthy':{
        'description':'This is a healthy green leaf.It does not contain any diseases.'
    }
}
    

model = torch.load('plant-disease-model-complete.pth')

def predict_image(img, model):
    """Converts image to array and return the predicted class
        with highest probability"""
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label

    return classes[preds[0].item()]

def predict(img_path):
    image = Image.open(img_path)
    transform = transforms.ToTensor()
    img_tensor = transform(image)
    prediction = str(predict_image(img_tensor, model))
    lower = prediction.lower()
    replace = lower.replace("___", "-")
    replace = replace.replace("_", "-")
    replace = " ".join(replace.split("-")[1:])
    return replace


def home_page(image_file):
    st.title('Plant Disease Detection Web')

    
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
     # Set the background image of the body element
        # Set the background image of the body element
    st.markdown(
        f"""
        <style>
        .stApp  {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-size: 100% 90%;
        }}
        </style>
        """,
        
        unsafe_allow_html=True
    )
    st.subheader('Protect your plants from all kind of diseases.Lack of immunity in crops has caused substantial increase in growth of crop diseases.Our Farmer Assisstant is here to help you out from crop demolition')
    st.code('''How do I help you?  
Utilizing advanced image processing and deep learning, our online service accurately detects the specific diseases affecting 
your plants, providing professional assistance for optimal plant health

Instant results    
Get results instantly without any long process. Just click the picture of leaf upload and get the results it’s that simple.''',language='python')
    uploaded_file = st.file_uploader('Upload an leaf image [ jpg , jpeg , png ]🔽', type=['jpg', 'jpeg', 'png'])
    

    if uploaded_file is not None:
        prediction = predict(uploaded_file)
        st.image(uploaded_file)
        st.warning("The predicted disease is: " + prediction)
        st.write("Description: ", disease_info[prediction]['description'])
        st.write("Symptoms: ", disease_info[prediction]['symptoms'])
        st.write("Diagnosis: ", disease_info[prediction]['diagnosis'])
         # display the disease video
        st.video(disease_info[prediction]['video_url'])
home_page('leaf.jpg')