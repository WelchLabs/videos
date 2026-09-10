def forward(self, x):

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        x = self.downsample(x)

    out += x
    out = self.relu(out)

    return out





