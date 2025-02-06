
import csv
import os.path as osp
from configparser import ConfigParser
from datetime import datetime
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from openpyxl import load_workbook
from skimage import io
from sklearn.metrics import confusion_matrix
from torch_geometric.loader import DataLoader, ImbalancedSampler

from Baselines.utils import *

config = ConfigParser()
RunCode = dates = datetime.now().strftime("%d-%m_%Hh%M")
project_root_dir = os.path.abspath(os.getcwd())

def create_config_file(type_dataset,type_graph,connectivity):
    configs_folder = osp.join(project_root_dir, f'results/{type_dataset}/{RunCode}')
    os.makedirs(configs_folder, exist_ok=True)
    config_filename = f"{configs_folder}/ConfigFile_{RunCode}.ini"
    graph_filename = f"{project_root_dir}/dataset/graphs/{type_graph}"
    os.makedirs(graph_filename, exist_ok=True)
    os.makedirs(f"{graph_filename}/{connectivity}", exist_ok=True)
    config["param"] = {
        'config_filename': config_filename,
        "type_dataset": type_dataset,
        'type_graph': type_graph,
        "graph_filename":graph_filename,
        "image_dataset_root": f"{project_root_dir}/dataset/images",
        "graph_dataset_folder": f"{graph_filename}/{connectivity}",
        "result_folder": f"{configs_folder}",
        "sigma":1.0,
        "threshold":0.01,
        "max_corners":500,
        "grid_size": 10
    }

    with open(config_filename, "w") as file:
        config.write(file)


def plot_image_with_nodes(img_path, data, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the image
    img = io.imread(img_path)

    # Initialize the graph
    G = nx.Graph()

    for i, feat in enumerate(data.x.cpu().numpy()):
        x=feat[0]
        y=feat[1]
        G.add_node(i, pos=(x, y))

    for edge in data.edge_index.t().cpu().numpy():
        G.add_edge(edge[0], edge[1])

    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(122)
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=False, node_size=10, node_color='r')
    plt.axis('off')

    # Extract the image name without extension and create the output path
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join(output_folder, img_name + '_graph.png')

    # Save the plot
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def save_plots(train_losses, metrics_dict):
    plt.figure(figsize=(12, 4))

    # Plot Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Metrics
    plt.subplot(1, 2, 2)
    metrics_names = list(metrics_dict.keys())
    metrics_values = list(metrics_dict.values())
    plt.bar(metrics_names, metrics_values)
    plt.title('Test Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Values')

    plt.tight_layout()
    plt.savefig(f'{config["param"]["result_folder"]}/training_plots.png')
    plt.show()

    # Save Training Evolution Data to CSV
    training_data = {'Iteration': list(range(1, len(train_losses) + 1)), 'Training Loss': train_losses}
    training_df = pd.DataFrame(training_data)
    training_df.to_csv(f"{config['param']['result_folder']}/training_evolution.csv", index=False)


def plot_and_save_training_performance(num_epochs, losses, folder_name):
    csv_file=f"{folder_name}/training_evolution.csv"
    pdf_file=f"{folder_name}/training_performance.pdf"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss'])
        epochs = range(1,num_epochs+1)
        for epoch, loss in zip(epochs, losses):
            writer.writerow([epoch, loss])

    plt.figure(figsize=(12, 5))
    # Plotting training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.tight_layout()

    plt.savefig(pdf_file, format='pdf')


def add_config(section_, key_, value_, ):
    if section_ not in list(config.sections()):
        config.add_section(section_)
    config[section_][key_] = str(value_)
    filename = config["param"]["config_filename"]
    with open(filename, "w") as conf:
        config.write(conf)


def shuffle_dataset(original_list):
    set_seed()
    shuffled_list = original_list.copy()
    random.shuffle(shuffled_list)
    return shuffled_list

def set_seed():
    # os.CUBLAS_WORKSPACE_CONFIG="4096:8"
    seed=42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(dataset_dir,batch_size=16,num_samples_per_class=0,type_data="train"):
        # Create datasets
        dataset = datasets.ImageFolder(dataset_dir, transform=transform(type_data=type_data))
        num_classes = len(dataset.classes)
        targets = [dataset.targets[i] for i in range(len(dataset))]
        indices = list(range(len(dataset)))


        # Create data loaders

        sampler = BalancedSampler(
            indices=indices,
            num_samples_per_class=num_samples_per_class,
            class_to_idx=dataset.class_to_idx,
            targets=targets
        )

        if type_data=="train":
            if num_samples_per_class==0:
                data_loader = DataLoader(dataset,  batch_size=batch_size,shuffle=True)
            else:
                data_loader = DataLoader(dataset,  batch_size=batch_size,  sampler=sampler)

        elif type_data=="test":
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        else:
            raise ValueError(f"Unsupported type_data: {type_data}. Use 'train' or 'test'.")

        print(f"Dataset details: {count_classes(data_loader)}")
        return num_classes, data_loader, dataset.classes


def transform(type_data="train"):
    # Common preprocessing steps
    preprocessing = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    if type_data == "train":
        # Additional augmentations for training
        augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(
                size=(224, 224),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BILINEAR
            )
        ]
        return transforms.Compose(augmentations + preprocessing)

    elif type_data == "test":
        # No augmentations for test data, only preprocessing
        return transforms.Compose(preprocessing)

    else:
        raise ValueError(f"Unsupported type_data: {type_data}. Use 'train' or 'test'.")



def save_training_details(train_loss, train_acc, test_acc, txt_file, pdf_file,model_name):
    """
    Save the provided metric values into a .txt file and generate plots saved into a PDF file.

    Args:
        train_loss (list): List of training loss values.
        train_acc (list): List of training accuracy values.
        test_acc (list): List of test accuracy values.
        txt_file (str): Path to the output .txt file for saving metrics.
        pdf_file (str): Path to the output PDF file for saving plots.
    """

    # Save the metrics into a .txt file
    with open(txt_file, "w") as f:
        f.write("Training Loss:\n")
        f.write(", ".join(map(str, train_loss)) + "\n\n")
        f.write("Training Accuracy:\n")
        f.write(", ".join(map(str, train_acc)) + "\n\n")
        f.write("Test Accuracy during training:\n")
        f.write(", ".join(map(str, test_acc)) + "\n\n")

    plt.figure(figsize=(10, 6))

    # Plot Train Loss
    plt.plot(range(len(train_loss)),train_loss, label='Train Loss', color='red', linestyle='--')

    # Plot Train Accuracy
    plt.plot(range(len(train_acc)), train_acc, label='Train Accuracy', color='blue')

    # Plot Test Accuracy
    plt.plot(range(len(test_acc)), test_acc, label='Test Accuracy', color='green')

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.title('Trainning stats for model:' + model_name )

    # Show legend
    plt.legend()

    # Add grid
    plt.grid(True)

    # Save the plot as a PDF file
    plt.savefig(pdf_file, format='pdf')

    # Show the plot
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, file_name="confusion_matrix.pdf"):
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Normalize the confusion matrix (optional, you can comment this line if you want raw counts)
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Create a heatmap using seaborn for better visualization
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(conf_matrix_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names,
                     yticklabels=class_names, cbar=True)

    # Add labels and title
    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)

    # Save the confusion matrix plot to a PDF file
    plt.savefig(file_name)
    plt.close()  # Close the plot after saving


def save_multiple_time_to_excel_with_date(cr, args):
    output_file_path = os.path.join(args.result_dir, f"result_for_{args.model_name}.xlsx")
    current_date = datetime.now().strftime("%Y-%m-%d")

    if os.path.exists(output_file_path):
        workbook = load_workbook(output_file_path)
        with pd.ExcelWriter(output_file_path, engine="openpyxl", mode="a") as writer:
            writer.book = workbook
            cr.to_excel(writer, sheet_name=current_date)
    else:
        cr.to_excel(output_file_path)

def graphdata_loader(graph_list,args,type_data="train"):
    set_seed()
      
    sampler=DistributedSampler(graph_list)
    if type_data == "train":
        dataset_loader = DataLoader(graph_list, batch_size=args.batch_size, sampler=sampler)
    else:
        dataset_loader = DataLoader(graph_list, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count())


    return dataset_loader

def Load_graphdata(dataset_source_path):
    set_seed()
    graph_list=[]
    label_dict={}
    assert os.path.isdir(dataset_source_path), "The provided dataset_source_path is not a valid directory."

    for file_name in os.listdir(dataset_source_path):

        data=torch.load(os.path.join(dataset_source_path,file_name))
        if data.y not in label_dict.keys():
            label_dict[data.y] = data.label_name
        graph_list.append(data)
    
    label_dict = dict(sorted(label_dict.items(), key=lambda item: int(item[0])))
    print("The dataset has been loaded. its contains: ",len(graph_list)," graphs.")
    print("graph 1:", graph_list[1])
    feat_size=data.x.shape[1]


    return graph_list,label_dict,feat_size, list(label_dict.values())