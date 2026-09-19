#pragma once
#include "scalergrad/engine.hpp"
#include <vector>
#include <random>
#include <stdexcept>

inline double random_uniform(double lo = -1.0, double hi = 1.0)
{
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(gen);
}

class Neuron
{
public:
    std::vector<ValuePtr> weight;
    ValuePtr bias;
    Neuron(int number_of_input)
    {
        for (int i = 0; i < number_of_input; i++)
            weight.push_back(make_value(random_uniform()));
        bias = make_value(random_uniform());
    }

    //? __call__ function
    ValuePtr operator()(const std::vector<ValuePtr> &input)
    {
        ValuePtr activation = bias;
        for (size_t i = 0; i < weight.size(); i++)
        {
            activation = activation + (weight[i] * input[i]);
        }
        return tanh(activation);
    };

    std::vector<ValuePtr> parameters()
    {
        std::vector<ValuePtr> params = weight;
        params.push_back(bias);
        return params;
    }
};

class Layer
{
public:
    std::vector<Neuron> neurons;

    Layer(int number_of_input, int number_of_neurons)
    {
        for (int i = 0; i < number_of_neurons; i++)
        {
            neurons.emplace_back(number_of_input);
        }
    }

    //? __call__ function
    std::vector<ValuePtr> operator()(const std::vector<ValuePtr> &x)
    {
        std::vector<ValuePtr> outs; //? Basically an array of tanh values from the neurons
        for (auto &n : neurons)
        {
            outs.push_back(n(x));
        }
        return outs;
    }

    std::vector<ValuePtr> parameters()
    {
        std::vector<ValuePtr> params;
        for (auto &n : neurons)
        {
            auto p = n.parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }
};

class MLP
{
public:
    std::vector<Layer> layers;

    MLP(int number_of_input, const std::vector<int> &number_of_neurons_per_layer)
    {
        std::vector<int> network_layer_sizes;
        network_layer_sizes.push_back(number_of_input);
        for (int n : number_of_neurons_per_layer)
            network_layer_sizes.push_back(n);

        for (size_t i = 0; i < number_of_neurons_per_layer.size(); i++)
            layers.emplace_back(network_layer_sizes[i], network_layer_sizes[i + 1]);
    }

private:
    //? Internal forward pass.
    //? Everything inside the network works with ValuePtr.
    std::vector<ValuePtr> forward(const std::vector<ValuePtr> &x)
    {
        std::vector<ValuePtr> output = x;
        for (auto &layer : layers)
            output = layer(output);
        return output;
    }

public:
    //__call__
    ValuePtr operator()(std::vector<double> x)
    {
        std::vector<ValuePtr> input;

        for (double value : x)
            input.push_back(make_value(value));

        std::vector<ValuePtr> output = forward(input);

        if (output.size() != 1)
        {
            throw std::runtime_error("MLP operator() expects exactly one output neuron");
        }
        return output[0];
    }

    //? For when client wants explicitly all outputs
    std::vector<ValuePtr> operator()(const std::vector<ValuePtr> &x)
    {
        return forward(x);
    }

    std::vector<ValuePtr> parameters()
    {
        std::vector<ValuePtr> params;
        for (auto &layer : layers)
        {
            auto p = layer.parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }
};