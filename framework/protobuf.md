# protobuf

## What's protocol buffers?

Protobuf is a data serializing protocol like a JSON or XML. But unlike them, the protobuf is not for humans, serialized data is compiled bytes and hard for the human reading.

```
It's description from Google official page:

Protocol buffers are Google's language-neutral, platform-neutral, extensible mechanism for serializing structured data – think XML, but smaller, faster, and simpler. You define how you want your data to be structured once, then you can use special generated source code to easily write and read your structured data to and from a variety of data streams and using a variety of languages.
```

## Why do we need another format for data serialization?

Modern server architectures is built on the constant communication of services. It can be REST API, GraphQl, RPC, Queues, etc. Services generate thousands of messages to each other, load the network and require a lot of resources. We need a fast way to serialize for transferring compact data between services.

A buffer can save us money and resources in the clouds like aws or gcloud.

Protobuf encoding is faster than json stream 2.3 times and json 2.7 times.

Protobuf decoding is faster than json stream 5.4 times and json 4.7 times.

## The problem

Why is it not so popular yet? Because protobuf is not so as simple as google says. Protobuf must preprocess from proto files to the sources of you programming language. Unfortunately, for some platforms, the protoc generator produces a very impractical code and it’s too hard to debug it. It’s not a human reading standard.