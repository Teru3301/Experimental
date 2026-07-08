
#include "model/perception/text_encoder.hpp"

#include "model/cognition/memory.hpp"
#include "model/cognition/world_model.hpp"

#include "model/expression/text_decoder.hpp"


namespace model
{
    
class model
{
private:
    //  perception
    text_encoder t_encoder;

    //  cognition
    memory memory;
    world_model brain;

    //  expression
    text_decoder t_decoder;

public:
    model();

    void load();
    void save();
    void create();

    void sense_text();

};

}

