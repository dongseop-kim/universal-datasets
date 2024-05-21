from univdt.components import MNIST


def test_mnist():
    root_dir = './data'
    trainset = MNIST(root_dir, split='train')
    assert len(trainset) == 60000
    testset = MNIST(root_dir, split='test')
    assert len(testset) == 10000

    data = trainset[0]
    image, label = data['image'], data['label']
    assert image.shape == (1, 28, 28)
    assert label.shape == (1,)
    assert label.item() == 5
