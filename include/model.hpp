
#pragma once

#include "model/perception/text.hpp"
#include "model/cognition/memory.hpp"
#include "model/cognition/world_model.hpp"
#include "model/expression/text.hpp"


namespace model
{
    
class model
{
private:
    //  perception
    perception::Text text_perception_;

    //  cognition
    memory memory_;
    world_model brain_;

    //  expression
    text_decoder t_decoder_;

public:
    model();

    void load();
    void save();
    void create();

    void sense_text(perception::TextInput input);

};


}

