#include <engine.hpp>

inline double random_uniform(double lo = -1.0, double hi = 1.0)
{
    static mt19937 gen(random_device{}());
    uniform_real_distribution<double> dist(lo, hi);
    return dist(gen);
}

class Neuron
{
public:
    vector<ValuePtr> weight;
    ValuePtr bias;
    Neuron(int number_of_input)
    {
        for (int i = 0; i < number_of_input; i++)
            weight.push_back(make_value(random_uniform()));
        bias = make_value(random_uniform());
    }

    //? __call__ function
    ValuePtr operator()(const vector<ValuePtr> &input)
    {
        ValuePtr activation = bias;
        for (size_t i = 0; i < weight.size(); i++)
        {
            activation = activation + (weight[i] * input[i]);
        }
        return tanh(activation);
    };

    vector<ValuePtr> parameters()
    {
        vector<ValuePtr> params = weight;
        params.push_back(bias);
        return params;
    }
};

class Layer
{
public:
    vector<Neuron> neurons;

    Layer(int number_of_input, int number_of_neurons)
    {
        for (int i = 0; i < number_of_neurons; i++)
        {
            neurons.emplace_back(number_of_input);
        }
    }

    //? __call__ function
    vector<ValuePtr> operator()(const vector<ValuePtr> &x)
    {
        vector<ValuePtr> outs; //? Basically an array of tanh values from the neurons
        for (auto &n : neurons)
        {
            outs.push_back(n(x));
        }
        return outs;
    }

    vector<ValuePtr> parameters()
    {
        vector<ValuePtr> params;
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
    vector<Layer> layers;

    MLP(int number_of_input, const vector<int> &number_of_neurons_per_layer)
    {
        vector<int> network_layer_sizes;
        network_layer_sizes.push_back(number_of_input);
        for (int n : number_of_neurons_per_layer)
            network_layer_sizes.push_back(n);

        for (size_t i = 0; i < number_of_neurons_per_layer.size(); i++)
            layers.emplace_back(network_layer_sizes[i], network_layer_sizes[i + 1]);
    }

private:
    //? Internal forward pass.
    //? Everything inside the network works with ValuePtr.
    vector<ValuePtr> forward(const vector<ValuePtr> &x)
    {
        vector<ValuePtr> output = x;
        for (auto &layer : layers)
            output = layer(output);
        return output;
    }

public:
    //__call__
    ValuePtr operator()(vector<double> x)
    {
        vector<ValuePtr> input;

        for (double value : x)
            input.push_back(make_value(value));

        vector<ValuePtr> output = forward(input);

        if (output.size() != 1)
        {
            throw runtime_error("MLP operator() expects exactly one output neuron");
        }
        return output[0];
    }

    //? For when client wants explicitly all outputs
    vector<ValuePtr> operator()(const vector<ValuePtr> &x)
    {
        return forward(x);
    }

    vector<ValuePtr> parameters()
    {
        vector<ValuePtr> params;
        for (auto &layer : layers)
        {
            auto p = layer.parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }
};
