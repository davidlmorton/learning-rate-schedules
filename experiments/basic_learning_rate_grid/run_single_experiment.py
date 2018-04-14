import matplotlib
matplotlib.use('Agg')

from sconce.data_generator import DataGenerator
from sconce.models import BasicClassifier
from sconce import rate_controllers
from sconce.trainers import ClassifierTrainer
from torch import optim
from torchvision import datasets

import click
import csv
import os
import random
import sconce
import torch
import uuid


@click.command()
@click.option('--initial-learning-rate', '-l', type=click.FLOAT, required=True)
@click.option('--experiment-directory', '-d', type=click.Path(), required=True)
@click.option('--yaml-file', '-y', type=click.File(), required=True)
@click.option('--rate-schedule', '-r',
        type=click.Choice(['piecewise-constant', 'clr-cosine',
            'clr-triangle', 'clr-step']),
        required=True)
@click.option('--batch-multiplier', '-m', type=click.INT, default=1)
@click.option('--batch-size', '-s', type=click.INT, default=500)
def single(initial_learning_rate, experiment_directory,
        yaml_file, rate_schedule, batch_size, batch_multiplier):
    os.makedirs(experiment_directory, exist_ok=True)

    model = BasicClassifier.new_from_yaml_file(yaml_file)

    training_generator = DataGenerator.from_pytorch(
            batch_size=batch_size,
            dataset_class=datasets.FashionMNIST)
    test_generator = DataGenerator.from_pytorch(
            batch_size=batch_size,
            dataset_class=datasets.FashionMNIST,
            train=False)

    if torch.cuda.is_available():
        model.cuda()
        training_generator.cuda()
        test_generator.cuda()

    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate,
            momentum=0.9, weight_decay=1e-4)

    trainer = ClassifierTrainer(model=model, optimizer=optimizer,
        training_data_generator=training_generator,
        test_data_generator=test_generator)

    rcs = {
        'clr-cosine': rate_controllers.CosineRateController(
            max_learning_rate=initial_learning_rate,
            min_learning_rate=initial_learning_rate/50),
        'clr-triangle': rate_controllers.TriangleRateController(
            max_learning_rate=initial_learning_rate,
            min_learning_rate=initial_learning_rate/50),
        'clr-step': rate_controllers.StepRateController(
            max_learning_rate=initial_learning_rate,
            min_learning_rate=initial_learning_rate/50,
            num_drops=1),
        'piecewise-constant': rate_controllers.ConstantRateController(
            learning_rate=initial_learning_rate,
            num_drops=3, movement_window=800, movement_threshold=0.20),
    }
    rate_controller = rcs[rate_schedule]

    if 'clr' in rate_schedule:
        trainer.multi_train(num_cycles=4, cycle_multiplier=1.5,
            rate_controller=rate_controller,
            batch_multiplier=batch_multiplier)
    else:
        trainer.train(num_epochs=20, rate_controller=rate_controller,
            batch_multiplier=batch_multiplier)

    fig = trainer.monitor.dataframe_monitor.plot(skip_first=100,
        smooth_window=30, metrics=['classification_accuracy', 'loss'])
    unique_part = str(uuid.uuid4())[-5:]
    ebs = batch_size * batch_multiplier
    name = f'{rate_schedule}-{initial_learning_rate}-{unique_part}_{ebs}'
    fig.savefig(os.path.join(experiment_directory,
        f'{name}-training-history.png'))

    monitor_file = os.path.join(experiment_directory, 'monitors.hd5')
    trainer.monitor.dataframe_monitor.save(monitor_file, key=name)

    acc = trainer.get_classification_accuracy()
    print(f"Classification Accuracy: {acc}")

    csv_file = os.path.join(experiment_directory, 'results.csv')
    write_header = not os.path.exists(csv_file)

    with open(os.path.join(experiment_directory, 'results.csv'), 'a') \
            as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        if write_header:
            writer.writerow(['rate_schedule', 'initial_learning_rate', 'acc', 'batch_size', 'batch_multiplier', 'effective_batch_size'])
        writer.writerow([rate_schedule, initial_learning_rate, acc, batch_size, batch_multiplier, ebs])


if __name__ == '__main__':
    single()
