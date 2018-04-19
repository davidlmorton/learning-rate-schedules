import matplotlib
matplotlib.use('Agg')

from sconce.data_generator import DataGenerator
from sconce.models import WideResnetImageClassifier
from sconce import rate_controllers
from sconce.trainers import ClassifierTrainer
from torch import optim
from torchvision import datasets, transforms

import click
import csv
import os
import random
import sconce
import torch
import uuid


@click.command()
@click.option('--max-learning-rate', '-l', type=click.FLOAT, required=True)
@click.option('--min-learning-rate', type=click.FLOAT, required=True)
@click.option('--experiment-directory', '-d', type=click.Path(), required=True)
@click.option('--dataset-class', '-c', type=click.Choice(['MNIST',
        'FashionMNIST', 'CIFAR10']), default='CIFAR10')
@click.option('--rate-schedule', '-r',
        type=click.Choice(['piecewise-constant', 'piecewise-constant-long',
            'triangle']),
        required=True)
@click.option('--batch-multiplier', '-m', type=click.INT, default=1)
@click.option('--batch-size', '-s', type=click.INT, default=100)
@click.option('--depth', '-t', type=click.INT, default=16)
@click.option('--widening-factor', '-f', type=click.INT, default=2)
@click.option('--num-epochs', '-n', type=click.INT, default=50)
def single(batch_multiplier, batch_size, dataset_class, depth,
        experiment_directory, max_learning_rate, min_learning_rate,
        num_epochs, rate_schedule, widening_factor):
    os.makedirs(experiment_directory, exist_ok=True)

    if dataset_class == 'CIFAR10':
        image_channels = 3
    else:
        image_channels = 1

    model = WideResnetImageClassifier(image_channels=image_channels,
            depth=depth, widening_factor=widening_factor)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(32, scale=(0.60, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.0,), std=(1.0,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.0,), std=(1.0,))
    ])

    training_generator = DataGenerator.from_pytorch(batch_size=batch_size,
        dataset_class=getattr(datasets, dataset_class),
        transform=transform)

    test_generator = DataGenerator.from_pytorch(batch_size=batch_size,
        dataset_class=getattr(datasets, dataset_class),
        transform=test_transform,
        train=False)

    if torch.cuda.is_available():
        model.cuda()
        training_generator.cuda()
        test_generator.cuda()

    optimizer = optim.SGD(model.parameters(), lr=1e-2,
            momentum=0.90, weight_decay=5e-4)

    trainer = ClassifierTrainer(model=model, optimizer=optimizer,
        training_data_generator=training_generator,
        test_data_generator=test_generator)

    rcs = {
        'triangle': rate_controllers.TriangleRateController(
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate),
        'piecewise-constant': rate_controllers.ConstantRateController(
            learning_rate=max_learning_rate,
            num_drops=3, movement_window=800//batch_multiplier, movement_threshold=0.15),
        'piecewise-constant-long': rate_controllers.ConstantRateController(
            learning_rate=max_learning_rate,
            num_drops=3, movement_window=8000//batch_multiplier, movement_threshold=0.15),
    }
    rate_controller = rcs[rate_schedule]

    trainer.train(num_epochs=num_epochs*batch_multiplier,
            rate_controller=rate_controller,
            batch_multiplier=batch_multiplier)

    num_steps = len(trainer.monitor.dataframe_monitor.df)
    num_epochs = num_steps / len(trainer.training_data_generator)

    fig = trainer.monitor.dataframe_monitor.plot(skip_first=500,
        smooth_window=200, metrics=['classification_accuracy', 'loss'])

    unique_part = str(uuid.uuid4())[-5:]
    effective_batch_size = batch_size * batch_multiplier

    name = f'{rate_schedule}-{max_learning_rate}-{unique_part}_{effective_batch_size}'
    fig.savefig(os.path.join(experiment_directory,
        f'{name}-training-history.png'))

    monitor_file = os.path.join(experiment_directory, 'monitors.hd5')
    trainer.monitor.dataframe_monitor.save(monitor_file, key=name)

    weights_file = os.path.join(experiment_directory, '%s.weights' % name)
    trainer.save_model_state(weights_file)

    classification_accuracy = trainer.get_classification_accuracy()
    print(f"Classification Accuracy: {classification_accuracy}")

    csv_file = os.path.join(experiment_directory, 'results.csv')
    write_header = not os.path.exists(csv_file)

    with open(os.path.join(experiment_directory, 'results.csv'), 'a') \
            as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        if write_header:
            writer.writerow([
                'batch_multiplier',
                'batch_size',
                'classification_accuracy',
                'dataset_class',
                'depth',
                'effective_batch_size',
                'max_learning_rate',
                'name',
                'num_epochs',
                'num_steps',
                'rate_schedule',
                'unique_part',
                'widening_factor'])
        writer.writerow([
            batch_multiplier,
            batch_size,
            classification_accuracy,
            dataset_class,
            depth,
            effective_batch_size,
            max_learning_rate,
            name,
            num_epochs,
            num_steps,
            rate_schedule,
            unique_part,
            widening_factor])


if __name__ == '__main__':
    single()
