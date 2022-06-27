
data = pd.read_csv('inputs/trainset.csv')
data=data.rename(columns={'Normal':'alines','>3 B-lines':'blines','Consolidation':'consolidation','Effusion':'effusion'})
data.image=data.image.apply(lambda x:x.replace(' ','_'))

cleaned = dict()
cleaned['image_name'] = []
cleaned['labels'] = []
cleaned['image_path'] = []

for label in os.listdir('dataset/clean_/'):
    for image in os.listdir('dataset/clean_/'+label):
        cleaned['image_name'].append(image)

    for _ in range(len(os.listdir('dataset/clean_/'+label))):
        cleaned['labels'].append(label)

    for path in os.listdir('dataset/clean_/'+label):
        cleaned['image_path'].append('clean_/'+label+'/'+path)
        
cleaned = pd.DataFrame(cleaned)
cleaned.labels=cleaned.labels.apply(lambda x:x.split('-'))

mlb = MultiLabelBinarizer()
binarized = mlb.fit_transform(cleaned.labels)
cleaned[mlb.classes_] = binarized

eff=data[data.effusion==1.0].reset_index()
eff = eff.loc[:,['image','alines','blines','consolidation','effusion']]
eff=eff.rename(columns={'image':'image_name'})

cons=data[data.consolidation==1.0].reset_index()
cons= cons.loc[:,['image','alines','blines','consolidation','effusion']]
cons=cons.rename(columns={'image':'image_name'})

cleaned=pd.concat([cleaned,eff,cons]).reset_index()





# new_cleaned=cleaned[cleaned.image_name.isin(data.image.values.tolist())]

# data=data[data.image.isin(cleaned.image_name.values.tolist())]

# new_data=data[['image','group_kfold']]

# data = pd.merge(new_data,new_cleaned,how='outer',left_on='image',right_on='image_name').drop(columns=['image_path','labels','image_name'])

print("data",data)

data.image=data.image.apply(lambda x: 'dataset/denoised_data/'+x)


train_df=data[data.group_kfold!=3]
val_df=data[data.group_kfold==3]
train_df=train_df.drop(columns=['group_kfold'])





def rename_labels1(label):
    if '-' in label:
        label=label.replace('-','')
    if '_' in label:
        label=label.replace('_','')
    if ' ' in label:
        label = label.replace(' ','')

    return label.lower()

def rename_labels2(label):
    if 'ablines' in label:
        label =label.replace('ablines','alines,blines')
    if 'a,blines' in label:
        label =label.replace('a,blines','alines,blines')
    if 'consolidations' in label:
        label = label.replace('consolidations','consolidation')
    
    return label


def combine_f(label):
    x = rename_labels1(label)
    x = rename_labels2(x)
    x =  x.split(',')
    return x

data1 = dict()
data1['image_name'] = []
data1['labels'] = []
data1['image_path'] = []

for label in os.listdir('dataset/Combined/'):
    for image in os.listdir('dataset/Combined/'+label):
        data1['image_name'].append(image)

    for _ in range(len(os.listdir('dataset/Combined/'+label))):
        data1['labels'].append(label)

    for path in os.listdir('dataset/Combined/'+label):
        data1['image_path'].append('Combined/'+label+'/'+path)

df2 = pd.DataFrame(data1)
# print(df2)
df2['pre_labels']=df2.labels.apply(combine_f)
mlb = MultiLabelBinarizer()
binarized = mlb.fit_transform(df2.pre_labels)
df2[mlb.classes_] = binarized
df2=df2.drop(columns=['image_name','labels','pre_labels'])
df2=df2.rename(columns={'image_path':'image'})
df2.image=df2.image.apply(lambda x:'dataset/'+x)

print("open",df2)

train_df=pd.concat([train_df,df2])
print("train_df\n",train_df)
train_df=train_df.reset_index().drop(columns=['index'])




# data = pd.read_csv('inputs/trainset.csv')
# data=data.rename(columns={'Normal':'alines','>3 B-lines':'blines','Consolidation':'consolidation','Effusion':'effusion'})
# data.image=data.image.apply(lambda x:x.replace(' ','_'))

# cleaned = dict()
# cleaned['image_name'] = []
# cleaned['labels'] = []
# cleaned['image_path'] = []

# for label in os.listdir('dataset/clean_/'):
#     for image in os.listdir('dataset/clean_/'+label):
#         cleaned['image_name'].append(image)

#     for _ in range(len(os.listdir('dataset/clean_/'+label))):
#         cleaned['labels'].append(label)

#     for path in os.listdir('dataset/clean_/'+label):
#         cleaned['image_path'].append('clean_/'+label+'/'+path)
        
# cleaned = pd.DataFrame(cleaned)
# cleaned.labels=cleaned.labels.apply(lambda x:x.split('-'))

# mlb = MultiLabelBinarizer()
# binarized = mlb.fit_transform(cleaned.labels)
# cleaned[mlb.classes_] = binarized

eff=data[data.effusion==1.0].reset_index()
eff = eff.loc[:,['image','alines','blines','consolidation','effusion']]
eff=eff.rename(columns={'image':'image_name'})

cons=data[data.consolidation==1.0].reset_index()
cons= cons.loc[:,['image','alines','blines','consolidation','effusion']]
cons=cons.rename(columns={'image':'image_name'})

cleaned=pd.concat([cleaned,eff,cons]).reset_index()

new_cleaned=cleaned[cleaned.image_name.isin(data.image.values.tolist())]

data=data[data.image.isin(cleaned.image_name.values.tolist())]


