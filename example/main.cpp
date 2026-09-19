#include <iostream>
#include "NeuralNet.hpp"
#include "visualization.hpp"

using namespace std;
int main()
{
    MLP n(3, {4, 4, 1});

    //? input data
    vector<vector<double>> xi = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, -1.0},
        {1.0, 1.0, -1.0},
    };

    //? target data
    vector<double> target = {1.0, -1.0, -1.0, 1.0};

    vector<ValuePtr> prediction;
    ValuePtr final_loss;

    for (int k = 0; k < 1000; k++)
    {
        //? Forward pass
        prediction.clear();
        for (const auto &x : xi)
            prediction.push_back(n(x));

        //? loss calculation
        ValuePtr loss = make_value(0.0, "loss");
        for (size_t i = 0; i < target.size(); i++)
        {
            ValuePtr difference = prediction[i] - target[i];
            loss = loss + pow(difference, 2);
        }

        //? zero grad
        for (auto &p : n.parameters())
            p->grad = 0.0;

        //? Backward pass
        backward(loss);

        //? Gradient descent
        for (auto &p : n.parameters())
        {
            p->data += -0.01 * p->grad;
        }

        if (k == 999)
        {
            final_loss = loss;
            final_loss->label = loss->label;
        }
    }
    cout << "Loss : " << final_loss->data << endl;
    cout << "Prediction data : " << "{ ";
    for (const auto &pred : prediction)
    {
        cout << pred->data << " ";
    }
    cout << " }" << endl;

    draw_dot(final_loss);

    return 0;
}