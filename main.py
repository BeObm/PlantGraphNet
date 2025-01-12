import torch.nn as nn
import torch.optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import argparse
from torch_geometric.nn import GCNConv, GraphConv, LEConv, TransformerConv, GATConv,SGConv
from model import *
import importlib


if __name__ == "__main__":
    set_seed()
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", help="Dataset name", default="train")
    parser.add_argument("--type_graph", default="harris", help="define how to construct nodes and egdes",
                       choices=["harris", "grid", "multi"])
    parser.add_argument("--hidden_dim", default=128, type=int, help="hidden_dim")
    parser.add_argument("--num_epochs", type=int, default=50, help="num_epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning_rate")
    parser.add_argument("--wd", type=float, default=0.005, help="wd")
    parser.add_argument("--Conv1", default=GraphConv, help="Conv1")
    parser.add_argument("--Conv2", default=GraphConv, help="Conv2")

    args = parser.parse_args()
    create_config_file(args.dataset,args.type_graph)
    # graph_constructor_obj = importlib.import_module(f"build_dataset.{args.type_graph}")
    # graph_constructor = getattr(graph_constructor_obj,"build_dataset")
    # # dataset_path = graph_constructor(config['param']["image_dataset_root"],config['param']["graph_dataset_name"])

    data = torch.load(config['param']["graph_dataset_name"])
    test_data = torch.load("dataset/graphs/val.pt")
    dataset= shuffle_dataset(data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dim = dataset[0].x.shape[1]
    hidden_dim = args.hidden_dim
    output_dim = 10
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    print(f"Feature size = {input_dim}|  num_class = {output_dim} ")

    model = GNNModel(input_dim, hidden_dim, output_dim, args.Conv1, args.Conv2).to(device)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,weight_decay=args.wd)
    pbar = tqdm(num_epochs)
    pbar.set_description("training model")
    best_loss=99999
    train_losses = []
    train_accuracies = []
    for epoch in range(num_epochs):
        loss, accuracy = train(model, train_loader, optimizer, criterion)
        train_losses.append(loss)
        train_accuracies.append(accuracy)
        if loss <= best_loss:
            best_loss = loss
            pbar.set_description(f"Training model.|Best loss={round(best_loss, 5)}")
        pbar.write(f'Epoch [{epoch}/{num_epochs}]: Loss: {round(loss, 5)}')
        pbar.update(1)
    plot_and_save_training_performance(num_epochs, train_losses, train_accuracies)
    test_metrics = test(model, test_loader)
    print("Test Metrics:")
    for metric, value in test_metrics.items():
        add_config("results", metric, value)
        print(f"{metric}: {value}")
