<template>
  <el-form-item :label="label">
    <el-input v-model="formData[name]" :placeholder="help" type="number" />
  </el-form-item>
</template>
<script>
import formFieldMixin from 'components/form-fields/form-field'

export default {
  mixins: [formFieldMixin]
}
</script>
<style lang="scss">
</style>
